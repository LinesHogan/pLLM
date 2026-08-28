#!/usr/bin/env python3
"""Unit tests for the worker-extension control plane (A2)."""

from __future__ import annotations

import json
import os
import unittest
from collections.abc import Mapping
from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock

import torch

from tllm import _core
from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.consumers.dummy.workload import DummyConfig
from tllm.consumers.dummy.workload import DummyWorkload
from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.consumer import ESampConsumer
from tllm.contracts.consumer_stats import TrainingStats, merge_training_stats_maps
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime import residual_runtime
from tllm.runtime.mp.control import (
    ControlPlane,
    TLLM_WORKER_EXTENSION_QUALNAME,
    WorkerShapeSpecializationReply,
    WorkerStatsReply,
    set_active_control_plane,
)
from tllm.runtime.stats import DistillerTimingStats, PathHotspotStats, PlacementStats
from tllm.runtime.mp.worker_extension import (
    TllmWorkerExtension,
    _runner_specialization_facts,
    _to_plain,
)
from tllm.runtime.workload_plugins import (
    NativeWorkload,
    NativeWorkloadResources,
    ReconstructedWorkload,
    WorkloadWorkerSpec,
    reconstruct_worker_bindings,
    static_backend_binding_manifest_wire,
)
from tllm.workload import Between, task, workload


@workload(id="mp.native-regions")
class _NativeMPWorkload:
    @task(timing=Between("model.execute", "step.end"))
    def run(self) -> None:
        return None


def _reconstruct_native_mp_workload(
    _config: object,
) -> ReconstructedWorkload:
    return ReconstructedWorkload(_NativeMPWorkload())


def _reconstruct_native_mp_workload_with_static_binding(
    _config: object,
) -> ReconstructedWorkload:
    return ReconstructedWorkload(
        _NativeMPWorkload(),
        specialization_key={"algorithm": "worker-only-static-binding"},
    )


@workload(id="status.alpha")
class _StatusAlphaWorkload(NativeWorkload):
    def __init__(self, metric: int) -> None:
        self._metric = metric
        self._resources = NativeWorkloadResources(
            workload=self,
            diagnostics_reader=lambda: {"metric": self._metric},
            worker_spec=WorkloadWorkerSpec(
                factory=f"{__name__}:_reconstruct_status_alpha_workload",
                config={"metric": metric},
            ),
        )

    def native_workload_resources(self) -> NativeWorkloadResources:
        return self._resources

    @task(timing=Between("model.execute", "step.end"))
    def run(self) -> None:
        return None


@workload(id="status.beta")
class _StatusBetaWorkload(NativeWorkload):
    def __init__(self, metric: int) -> None:
        self._metric = metric
        self._resources = NativeWorkloadResources(
            workload=self,
            diagnostics_reader=lambda: {"metric": self._metric},
            worker_spec=WorkloadWorkerSpec(
                factory=f"{__name__}:_reconstruct_status_beta_workload",
                config={"metric": metric},
            ),
        )

    def native_workload_resources(self) -> NativeWorkloadResources:
        return self._resources

    @task(timing=Between("model.execute", "step.end"))
    def run(self) -> None:
        return None


def _status_metric(raw: Mapping[str, object]) -> int:
    metric = raw.get("metric")
    if type(metric) is not int:
        raise TypeError("status workload metric must be an integer")
    return metric


def _reconstruct_status_alpha_workload(
    raw: Mapping[str, object],
) -> NativeWorkloadResources:
    return _StatusAlphaWorkload(_status_metric(raw)).native_workload_resources()


def _reconstruct_status_beta_workload(
    raw: Mapping[str, object],
) -> NativeWorkloadResources:
    return _StatusBetaWorkload(_status_metric(raw)).native_workload_resources()


class _HookModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        body = torch.nn.Module()
        body.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.model = body
        self.config = SimpleNamespace(hidden_size=4)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _FakeEngineCore:
    """Stands in for the EngineCore client; records collective_rpc calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, float | None]] = []
        self.handler = TllmWorkerExtension()
        self.handlers = [self.handler]
        self.fail_method: str | None = None
        self.fail_handler_index: int | None = None
        self.single_timeline_reply = False
        self.reply_limits: dict[str, int] = {}

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls.append((str(method), tuple(args), timeout))
        replies = []
        for index, handler in enumerate(self.handlers):
            if self.single_timeline_reply and str(method) == "tllm_workload_timeline" and index:
                continue
            limit = self.reply_limits.get(str(method))
            if limit is not None and index >= limit:
                continue
            if self.fail_method == str(method) and self.fail_handler_index == index:
                raise RuntimeError(f"simulated worker {index} failure for {method}")
            replies.append(getattr(handler, str(method))(*args))
        return replies


class _FakeLLM:
    def __init__(self) -> None:
        self.llm_engine = type("Engine", (), {"engine_core": _FakeEngineCore()})()


class _CoercibleInteger:
    def __int__(self) -> int:
        return 1


class _CoercibleText:
    def __init__(self, value: str) -> None:
        self._value = value

    def __str__(self) -> str:
        return self._value


def _shape_specialization_request(
    signature: str, plan_epoch: int
):
    from tllm.contracts.shape_specialization import (
        RegionShapeSpecializationInput,
        RegionShapeSpecializationRequest,
    )

    return RegionShapeSpecializationRequest(
        base_cache_key="a" * 64,
        base_signature=signature,
        observed_plan_epoch=plan_epoch,
        region_id="region-0",
        artifact_key="b" * 64,
        inputs=(
            RegionShapeSpecializationInput(
                binding_id="hidden", shape=(2, 4), dtype="torch.float32", device="cpu"
            ),
        ),
    )


def _shape_specialization_state(signature: str, plan_epoch: int):
    from tllm.runtime.state import ResidualRuntimeState
    from tllm.runtime.workload_execution import NativeWorkloadExecution

    compilation = SimpleNamespace(
        decision=SimpleNamespace(signature=signature),
        physical=SimpleNamespace(plan_epoch=plan_epoch),
    )
    execution = NativeWorkloadExecution(
        compilation=compilation,  # type: ignore[arg-type]
        executor=SimpleNamespace(),  # type: ignore[arg-type]
        completion_anchor="step.end",
        _owned_consumers=(),
    )
    return (
        ResidualRuntimeState(
            workload_compilation=compilation,  # type: ignore[arg-type]
            workload_execution=execution,
        ),
        execution,
    )


class ControlPlaneUnitTest(unittest.TestCase):
    def test_workload_worker_spec_wire_is_closed_and_does_not_coerce(self) -> None:
        self.assertEqual(
            WorkloadWorkerSpec.from_wire(
                {"factory": "test.module:factory", "config": {"value": 1}}
            ).to_wire(),
            {"factory": "test.module:factory", "config": {"value": 1}},
        )
        for payload in (
            {"factory": "test.module:factory"},
            {"factory": "test.module:factory", "config": {}, "extra": True},
            {"factory": 7, "config": {}},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises((TypeError, ValueError)):
                    WorkloadWorkerSpec.from_wire(payload)  # type: ignore[arg-type]

    def test_worker_timeline_reports_json_specialization_facts(self) -> None:
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.graph_policy import GraphMode
        from tllm.workgraph.anchors import BackendResolution

        extension = TllmWorkerExtension()
        extension.model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            max_num_reqs=8,
            max_num_tokens=16,
            model_config=SimpleNamespace(
                dtype=torch.float32,
                get_hidden_size=lambda: 64,
            ),
        )
        saved = residual_runtime.RUNTIME.workgraph_backend_resolution
        try:
            residual_runtime.RUNTIME.workgraph_backend_resolution = BackendResolution(
                capabilities=V027MRv2Driver().capabilities(),
                graph_mode=GraphMode.EAGER,
            )

            payload = extension.tllm_workload_timeline(configured_budget=64)
        finally:
            residual_runtime.RUNTIME.workgraph_backend_resolution = saved

        self.assertIn("timeline", payload)
        self.assertEqual(
            payload["specialization"],
            {
                "device": "cpu",
                "dtype": "torch.float32",
                "hidden_size": 64,
                "max_prefill_tokens": 16,
                "rows": 8,
            },
        )

    def test_runner_specialization_facts_reject_non_exact_positive_integers(self) -> None:
        def runner(
            rows: object, hidden_size: object, max_prefill_tokens: object = 16
        ) -> SimpleNamespace:
            return SimpleNamespace(
                device=torch.device("cpu"),
                max_num_reqs=rows,
                max_num_tokens=max_prefill_tokens,
                model_config=SimpleNamespace(
                    dtype=torch.float32,
                    get_hidden_size=lambda: hidden_size,
                ),
            )

        self.assertIsNotNone(_runner_specialization_facts(runner(2, 4)))
        for rows, hidden_size, max_prefill_tokens in (
            (True, 4, 16),
            (2.0, 4, 16),
            ("2", 4, 16),
            (_CoercibleInteger(), 4, 16),
            (2, True, 16),
            (2, 4.0, 16),
            (2, "4", 16),
            (2, _CoercibleInteger(), 16),
            (0, 4, 16),
            (2, 0, 16),
            (2, 4, True),
            (2, 4, 0),
            (2, 4, "16"),
        ):
            with self.subTest(
                rows=rows,
                hidden_size=hidden_size,
                max_prefill_tokens=max_prefill_tokens,
            ):
                with self.assertRaises((TypeError, ValueError)):
                    _runner_specialization_facts(
                        runner(rows, hidden_size, max_prefill_tokens)
                    )

        fallback = runner(2, 4)
        del fallback.max_num_tokens
        fallback.max_num_batched_tokens = 16
        facts = _runner_specialization_facts(fallback)
        assert facts is not None
        self.assertEqual(facts.max_prefill_tokens, 16)

        missing = runner(2, 4)
        del missing.max_num_tokens
        with self.assertRaisesRegex(RuntimeError, "max_num_tokens"):
            _runner_specialization_facts(missing)

    def test_runner_specialization_layer_count_comes_from_bound_driver(self) -> None:
        from tllm.backends.protocol import RunnerBinding

        runner = SimpleNamespace(
            device=torch.device("cpu"),
            max_num_reqs=2,
            max_num_tokens=16,
            model_config=SimpleNamespace(
                dtype=torch.float32,
                get_hidden_size=lambda: 4,
            ),
        )
        driver = mock.Mock()
        driver.runner_model_layer_count.return_value = 7
        setattr(
            runner,
            "_tllm_runner_binding",
            RunnerBinding(
                driver=driver,
                runner=runner,
                capabilities=mock.Mock(),
            ),
        )

        facts = _runner_specialization_facts(runner)

        self.assertIsNotNone(facts)
        assert facts is not None
        self.assertEqual(facts.model_layer_count, 7)
        driver.runner_model_layer_count.assert_called_once_with(runner)

    def test_rust_specialization_normalizer_changes_signed_compile_key(self) -> None:
        from tllm.runtime.runtime import TllmRuntime
        from tllm.schedule import CompileOptions

        runtime = TllmRuntime(
            workloads=(_NativeMPWorkload(),), workload_memory_budget_bytes=64
        )
        before = runtime.workload_compile_options

        normalized = _core.normalize_runtime_specialization_options(
            before.to_json(),
            '{"device":"cuda:0","dtype":"torch.bfloat16",'
            '"hidden_size":896,"rows":32}',
        )
        after = CompileOptions.from_json(normalized)
        self.assertNotEqual(before.to_json(), after.to_json())
        self.assertIn(("runtime/hidden_size", "896"), after.shape_specialization)

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
        saved = (
            residual_runtime.RUNTIME.consumer,
            residual_runtime.RUNTIME.workload_execution,
            residual_runtime.RUNTIME.workload_execution_store,
        )
        try:
            residual_runtime.RUNTIME.workload_execution = None
            residual_runtime.RUNTIME.workload_execution_store = None
            plane.apply_enable_flags({"esamp_training_enabled": True})
            with self.assertRaises(ValueError):
                plane.apply_enable_flags({"nonsense_flag": True})
        finally:
            (
                residual_runtime.RUNTIME.consumer,
                residual_runtime.RUNTIME.workload_execution,
                residual_runtime.RUNTIME.workload_execution_store,
            ) = saved

    def test_worker_enable_flags_reject_a_live_native_epoch(self) -> None:
        from tllm.runtime.state import ResidualRuntimeState

        llm = _FakeLLM()
        plane = ControlPlane(llm)
        state = ResidualRuntimeState(workload_execution=object())

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            with self.assertRaisesRegex(RuntimeError, "coordinator-signed"):
                plane.apply_enable_flags({"esamp_training_enabled": False})

    def test_mp_live_native_enable_transition_fails_before_state_reset(self) -> None:
        from tllm.backends.timeline import export_main_workload_timeline
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.specialization import RuntimeSpecializationSpec

        from tllm.consumers.esamp.region_workload import (
            ESampConfig,
            ESampWorkload,
        )

        workload = ESampWorkload(ESampConfig(enabled=True))
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
        )
        initial = runtime.compile_workloads(
            export_main_workload_timeline(
                V027MRv2Driver().capabilities(),
                graph_mode="eager",
                static_headroom_bytes=1 << 20,
            )
        )
        model = workload.distiller
        optimizer = workload.optimizer
        state_version = workload.state_version

        with self.assertRaisesRegex(RuntimeError, "state migration"):
            runtime.transition_native_workload_enabled(False, plane=object())

        self.assertIs(runtime.workload_compilation, initial)
        self.assertTrue(workload.config.enabled)
        self.assertIs(workload.distiller, model)
        self.assertIs(workload.optimizer, optimizer)
        self.assertEqual(workload.state_version, state_version)

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

    def test_parent_rejects_legacy_workload_without_an_immutable_worker_spec(self) -> None:
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.graph_policy import GraphMode
        from tllm.runtime.runtime import TllmRuntime
        from tllm.workgraph.anchors import BackendResolution
        from tllm.workload import Between, task, workload

        legacy_adapter = object()

        @workload(id="mp.compiler")
        class MPWorkload:
            @task(timing=Between("model.execute", "step.end"))
            def run(self) -> None:
                return None

            @property
            def legacy_consumer(self) -> object:
                return legacy_adapter

        runtime = TllmRuntime(
            workloads=(MPWorkload(),), workload_memory_budget_bytes=64
        )
        llm = _FakeLLM()
        llm.llm_engine.engine_core.handler.model_runner = SimpleNamespace(
            device="cpu"
        )
        plane = ControlPlane(llm)
        saved_resolution = residual_runtime.RUNTIME.workgraph_backend_resolution
        saved_compilation = residual_runtime.RUNTIME.workload_compilation
        try:
            residual_runtime.RUNTIME.workgraph_backend_resolution = BackendResolution(
                capabilities=V027MRv2Driver().capabilities(),
                graph_mode=GraphMode.EAGER,
            )
            with self.assertRaisesRegex(
                RuntimeError, "explicit worker reconstruction spec"
            ):
                runtime._coordinate_worker_compilation(plane)
            self.assertEqual(llm.llm_engine.engine_core.calls, [])
            self.assertIsNone(runtime.workload_compilation)
            self.assertIs(
                residual_runtime.RUNTIME.workload_compilation, saved_compilation
            )
        finally:
            residual_runtime.RUNTIME.workgraph_backend_resolution = saved_resolution
            residual_runtime.RUNTIME.workload_compilation = saved_compilation

    def test_native_worker_reconstructs_and_reports_exact_region_artifacts(self) -> None:
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.graph_policy import GraphMode
        from tllm.runtime.runtime import TllmRuntime
        from tllm.workgraph.anchors import BackendResolution

        spec = WorkloadWorkerSpec(
            f"{__name__}:_reconstruct_native_mp_workload",
            {},
        )
        runtime = TllmRuntime(
            workloads=(_NativeMPWorkload(),),
            workload_memory_budget_bytes=64,
            workload_worker_specs=(spec,),
        )
        llm = _FakeLLM()
        llm.llm_engine.engine_core.handler.model_runner = SimpleNamespace(
            device="cpu"
        )
        plane = ControlPlane(llm)
        saved_resolution = residual_runtime.RUNTIME.workgraph_backend_resolution
        saved_compilation = residual_runtime.RUNTIME.workload_compilation
        saved_execution = residual_runtime.RUNTIME.workload_execution
        saved_execution_store = residual_runtime.RUNTIME.workload_execution_store
        saved_physical = residual_runtime.RUNTIME.physical_executor
        try:
            residual_runtime.RUNTIME.workload_execution_store = None
            residual_runtime.RUNTIME.workgraph_backend_resolution = BackendResolution(
                capabilities=V027MRv2Driver().capabilities(),
                graph_mode=GraphMode.EAGER,
            )
            runtime._coordinate_worker_compilation(plane)

            compilation = runtime.workload_compilation
            self.assertIsNotNone(compilation)
            execution = residual_runtime.RUNTIME.workload_execution
            self.assertIsNotNone(execution)
            assert compilation is not None and execution is not None
            self.assertEqual(
                execution.artifact_report,
                tuple(
                    (region.region_id, region.artifact_key)
                    for region in compilation.decision.execution_regions
                ),
            )
            prepare_method, prepare_args, _timeout = llm.llm_engine.engine_core.calls[-4]
            self.assertEqual(prepare_method, "tllm_prepare_workload_decision")
            self.assertEqual(prepare_args[-1], [spec.to_wire()])
            commit_method, commit_args, _timeout = llm.llm_engine.engine_core.calls[-3]
            self.assertEqual(commit_method, "tllm_commit_workload_decision")
            self.assertEqual(commit_args[0], compilation.decision.signature)
            activation_method, activation_args, _timeout = llm.llm_engine.engine_core.calls[-2]
            self.assertEqual(activation_method, "tllm_activate_workload_decision")
            self.assertEqual(activation_args, (
                compilation.decision.signature,
                compilation.physical.plan_epoch,
            ))
            self.assertEqual(
                llm.llm_engine.engine_core.calls[-1][0],
                "tllm_finalize_workload_decision",
            )
            statuses = plane.workload_status()
            self.assertEqual(len(statuses), 1)
            self.assertEqual(statuses[0]["pid"], os.getpid())
            self.assertEqual(
                statuses[0]["signature"], compilation.decision.signature
            )
            self.assertEqual(
                statuses[0]["plan_epoch"], compilation.physical.plan_epoch
            )
            self.assertEqual(
                statuses[0]["artifacts"],
                [
                    {"region_id": region_id, "artifact_key": artifact_key}
                    for region_id, artifact_key in execution.artifact_report
                ],
            )
            self.assertEqual(statuses[0]["state_versions"], [])
        finally:
            current = residual_runtime.RUNTIME.workload_execution
            if current is not None and current is not saved_execution:
                current.close()
            residual_runtime.RUNTIME.workgraph_backend_resolution = saved_resolution
            residual_runtime.RUNTIME.workload_compilation = saved_compilation
            residual_runtime.RUNTIME.workload_execution = saved_execution
            residual_runtime.RUNTIME.workload_execution_store = saved_execution_store
            residual_runtime.RUNTIME.physical_executor = saved_physical

    def test_parent_cold_compiles_recirculation_bundle_before_load_only_worker_reconstruction(
        self,
    ) -> None:
        """The EngineCore worker receives a signed bundle, never a compiler callback."""

        from tllm.backends.protocol import RunnerBinding
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.consumers.recirculation import (
            RecirculationConfig,
            VllmRecirculationRegionWorkload,
        )
        from tllm.consumers.recirculation import vllm_region_workload
        from tllm.runtime.graph_policy import GraphMode
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.state import ResidualRuntimeState
        from tllm.workgraph.anchors import BackendResolution

        driver = V027MRv2Driver()
        workload = VllmRecirculationRegionWorkload(
            RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.2)
        )
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        llm = _FakeLLM()
        runner = SimpleNamespace(
            device=torch.device("cpu"),
            max_num_reqs=2,
            max_num_tokens=2,
            model_config=SimpleNamespace(
                dtype=torch.float32,
                get_hidden_size=lambda: 4,
            ),
        )
        layer_driver = mock.Mock()
        layer_driver.runner_model_layer_count.return_value = 3
        setattr(
            runner,
            "_tllm_runner_binding",
            RunnerBinding(
                driver=layer_driver,
                runner=runner,
                capabilities=driver.capabilities(),
            ),
        )
        llm.llm_engine.engine_core.handler.model_runner = runner
        plane = ControlPlane(llm)
        state = ResidualRuntimeState(
            workgraph_backend_resolution=BackendResolution(
                capabilities=driver.capabilities(),
                graph_mode=GraphMode.EAGER,
            )
        )
        compile_stage = _core.compile_typed_recurrence_stage_artifact_json
        compile_bundle = _core.compile_typed_recurrence_stage_artifact_bundle_json
        build_graph = vllm_region_workload.build_recirculation_graph

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch.object(
                _core,
                "compile_typed_recurrence_stage_artifact_json",
                wraps=compile_stage,
            ) as artifact_compile,
            mock.patch.object(
                _core,
                "compile_typed_recurrence_stage_artifact_bundle_json",
                wraps=compile_bundle,
            ) as bundle_compile,
            mock.patch.object(
                vllm_region_workload,
                "build_recirculation_graph",
                wraps=build_graph,
            ) as graph_compile,
        ):
            runtime._coordinate_worker_compilation(plane)

        resources = workload.native_workload_resources()
        worker_spec = resources.worker_spec
        assert worker_spec is not None
        assert "typed_stage_artifact_bundle" in worker_spec.config
        assert artifact_compile.call_count == 2
        assert bundle_compile.call_count == 1
        assert graph_compile.call_count == 2
        self.assertEqual(
            llm.llm_engine.engine_core.calls[-4][0],
            "tllm_prepare_workload_decision",
        )
        worker_wire = llm.llm_engine.engine_core.calls[-4][1][-1]
        self.assertEqual(worker_wire, [worker_spec.to_wire()])

    def test_caller_supplied_worker_spec_without_parent_resource_is_closed(self) -> None:
        from tllm.backends.timeline import export_main_workload_timeline
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.runtime import TllmRuntime

        timeline = export_main_workload_timeline(
            V027MRv2Driver().capabilities(),
            graph_mode="eager",
            static_headroom_bytes=64,
        )
        spec = WorkloadWorkerSpec(
            f"{__name__}:_reconstruct_native_mp_workload", {}
        )
        runtime = TllmRuntime(
            workloads=(_NativeMPWorkload(),),
            workload_memory_budget_bytes=64,
            workload_worker_specs=(spec,),
        )
        compilation = runtime.compile_workloads(timeline)
        bindings = runtime.workload_execution_bindings
        assert bindings is not None

        expected_manifest = json.dumps(
            [
                {
                    "workload_id": "mp.native-regions",
                    "factory": spec.factory,
                    "config": {},
                    "specialization_key": [],
                    "backend_region_implementation_ids": [],
                }
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertEqual(
            compilation.options.static_binding_manifest, expected_manifest
        )
        self.assertIsNone(bindings.static_binding_entries[0][1])
        self.assertEqual(
            static_backend_binding_manifest_wire(bindings.static_binding_entries),
            expected_manifest,
        )
        reconstructed = reconstruct_worker_bindings(compilation, (spec,))
        self.assertEqual(
            static_backend_binding_manifest_wire(
                reconstructed.static_binding_entries
            ),
            expected_manifest,
        )

        tampered_spec = WorkloadWorkerSpec(
            f"{__name__}:_reconstruct_native_mp_workload_with_static_binding",
            {},
        )
        with (
            mock.patch(
                "tllm.runtime.workload_plugins._resolve_factory"
            ) as resolve_factory,
            self.assertRaisesRegex(RuntimeError, "parent-signed static binding"),
        ):
            reconstruct_worker_bindings(compilation, (tampered_spec,))
        resolve_factory.assert_not_called()

        worker_static_spec = WorkloadWorkerSpec(
            f"{__name__}:_reconstruct_native_mp_workload_with_static_binding",
            {},
        )
        drifted_runtime = TllmRuntime(
            workloads=(_NativeMPWorkload(),),
            workload_memory_budget_bytes=64,
            workload_worker_specs=(worker_static_spec,),
        )
        drifted_compilation = drifted_runtime.compile_workloads(timeline)
        with self.assertRaisesRegex(RuntimeError, "static backend binding"):
            reconstruct_worker_bindings(drifted_compilation, (worker_static_spec,))

    def test_first_epoch_commit_stays_invisible_until_activation(self) -> None:
        from tllm.backends.timeline import export_main_workload_timeline
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.state import ResidualRuntimeState

        runtime = TllmRuntime(
            workloads=(_NativeMPWorkload(),), workload_memory_budget_bytes=64
        )
        compilation = runtime.compile_workloads(
            export_main_workload_timeline(
                V027MRv2Driver().capabilities(),
                graph_mode="eager",
                static_headroom_bytes=64,
            )
        )
        state = ResidualRuntimeState()
        previous_compilation = object()
        previous_execution = mock.Mock()
        state.workload_compilation = previous_compilation
        state.workload_execution = previous_execution  # type: ignore[assignment]
        state.physical_executor = previous_execution  # type: ignore[assignment]
        extension = TllmWorkerExtension()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            report = extension.tllm_prepare_workload_decision(
                compilation.cache_key,
                compilation.graph.to_json(),
                compilation.timeline.to_json(),
                compilation.options.to_json(),
                compilation.decision.to_json(),
                compilation.physical.plan_epoch,
            )
            self.assertFalse(report["committed"])
            self.assertIs(state.workload_compilation, previous_compilation)
            self.assertIs(state.workload_execution, previous_execution)

            extension.tllm_commit_workload_decision(
                compilation.decision.signature, compilation.physical.plan_epoch
            )
            self.assertIs(state.workload_compilation, previous_compilation)
            self.assertIs(state.workload_execution, previous_execution)
            extension.tllm_activate_workload_decision(
                compilation.decision.signature, compilation.physical.plan_epoch
            )
            self.assertEqual(
                state.workload_compilation.decision.signature,  # type: ignore[union-attr]
                compilation.decision.signature,
            )
            with self.assertRaisesRegex(RuntimeError, "after global activation"):
                extension.tllm_abort_workload_decision(
                    compilation.decision.signature, compilation.physical.plan_epoch
                )
            extension.tllm_finalize_workload_decision(
                compilation.decision.signature, compilation.physical.plan_epoch
            )

        self.assertEqual(
            state.workload_compilation.decision.signature,  # type: ignore[union-attr]
            compilation.decision.signature,
        )

    def test_prepare_workload_decision_rejects_scalar_coercion(self) -> None:
        extension = TllmWorkerExtension()
        wire: dict[str, object] = {
            "cache_key": "a" * 64,
            "graph_json": "{}",
            "timeline_json": "{}",
            "options_json": "{}",
            "decision_json": "{}",
            "plan_epoch": 1,
        }

        with mock.patch("tllm.schedule.install_workload_compilation") as install:
            for field in (
                "cache_key",
                "graph_json",
                "timeline_json",
                "options_json",
                "decision_json",
            ):
                with self.subTest(field=field):
                    invalid = dict(wire)
                    invalid[field] = object()
                    with self.assertRaisesRegex(TypeError, f"{field} must be a string"):
                        extension.tllm_prepare_workload_decision(**invalid)  # type: ignore[arg-type]
                    install.assert_not_called()
            for invalid_epoch in (True, 0, "1"):
                with self.subTest(plan_epoch=invalid_epoch):
                    invalid = dict(wire)
                    invalid["plan_epoch"] = invalid_epoch
                    with self.assertRaisesRegex(TypeError, "plan_epoch must be a positive integer"):
                        extension.tllm_prepare_workload_decision(**invalid)  # type: ignore[arg-type]
                    install.assert_not_called()

    def test_all_epoch_phase_rpcs_reject_coerced_identities(self) -> None:
        signature = "a" * 64
        extension = TllmWorkerExtension()
        extension._tllm_staged_workload = SimpleNamespace(
            signature=signature,
            plan_epoch=1,
        )
        methods = (
            extension.tllm_commit_workload_decision,
            extension.tllm_abort_workload_decision,
            extension.tllm_finalize_workload_decision,
            extension.tllm_drain_workload_epochs,
        )
        for method in methods:
            for invalid_signature, invalid_epoch in (
                (_CoercibleText(signature), 1),
                (signature, True),
                (signature, 1.0),
                (signature, "1"),
                (signature, _CoercibleInteger()),
            ):
                with self.subTest(method=method.__name__, value=(invalid_signature, invalid_epoch)):
                    with self.assertRaises((TypeError, ValueError)):
                        method(invalid_signature, invalid_epoch)  # type: ignore[arg-type]

    def test_parent_rejects_coerced_native_prepare_report(self) -> None:
        from tllm.backends.timeline import export_main_workload_timeline
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.runtime import TllmRuntime

        runtime = TllmRuntime(
            workloads=(_NativeMPWorkload(),),
            workload_memory_budget_bytes=1 << 20,
            workload_worker_specs=(
                WorkloadWorkerSpec(f"{__name__}:_reconstruct_native_mp_workload", {}),
            ),
        )
        compilation = runtime.compile_workloads(
            export_main_workload_timeline(
                V027MRv2Driver().capabilities(),
                graph_mode="eager",
                static_headroom_bytes=1 << 20,
            )
        )
        region = compilation.decision.execution_regions[0]

        class _Plane:
            def __init__(self, report: dict[str, object]) -> None:
                self._report = report

            def prepare_workload_compilation(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
                return [self._report]

            def abort_workload_compilation(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
                return []

        standard = {
            "signature": compilation.decision.signature,
            "plan_epoch": compilation.physical.plan_epoch,
            "artifacts": [
                {
                    "region_id": region.region_id,
                    "artifact_key": region.artifact_key,
                }
            ],
            "native": True,
            "committed": False,
        }
        malformed_reports = (
            {**standard, "native": 1},
            {
                **standard,
                "artifacts": [
                    {
                        "region_id": _CoercibleText(region.region_id),
                        "artifact_key": _CoercibleText(region.artifact_key),
                    }
                ],
            },
            {**standard, "committed": 0},
        )
        for report in malformed_reports:
            with self.subTest(report=report):
                with self.assertRaisesRegex(TypeError, "workload-install"):
                    runtime._install_workload_compilation_on_workers(
                        _Plane(report), compilation
                    )

    def test_abort_restores_native_hidden_hook_topology_and_resources(self) -> None:
        from tllm.backends.timeline import export_main_workload_timeline
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.consumers.esamp.region_workload import (
            ESampConfig,
            ESampWorkload,
        )
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.specialization import RuntimeSpecializationSpec
        from tllm.runtime.state import ResidualRuntimeState

        workload_instance = ESampWorkload(
            ESampConfig(
                source_layer_path="model.layers.0",
                target_layer_path="model.layers.0",
            )
        )
        runtime = TllmRuntime(
            workloads=(workload_instance,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(
                2,
                4,
                torch.float32,
                torch.device("cpu"),
            )
        )
        compilation = runtime.compile_workloads(
            export_main_workload_timeline(
                V027MRv2Driver().capabilities(),
                graph_mode="eager",
                static_headroom_bytes=1 << 20,
            )
        )
        worker_spec = workload_instance.native_workload_resources().worker_spec
        assert worker_spec is not None

        model = _HookModel()
        layer = model.model.layers[0]
        original_forward = layer.forward
        original_forward_function = original_forward.__func__
        original_compute_logits = model.compute_logits
        runner = SimpleNamespace(
            device=torch.device("cpu"),
            max_num_reqs=2,
            max_num_tokens=16,
            model_config=SimpleNamespace(
                dtype=torch.float32,
                get_hidden_size=lambda: 4,
            ),
            model=model,
            sampler=SimpleNamespace(),
        )
        extension = TllmWorkerExtension()
        extension.model_runner = runner
        state = ResidualRuntimeState()
        previous_compilation = object()
        previous_execution = mock.Mock()
        previous_execution.sampler_modifier_provider.return_value = None
        previous_decode_rows = torch.tensor([7], dtype=torch.long)
        state.workload_compilation = previous_compilation
        state.workload_execution = previous_execution  # type: ignore[assignment]
        state.physical_executor = previous_execution  # type: ignore[assignment]
        state.decode_row_idx = previous_decode_rows

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            extension.tllm_prepare_workload_decision(
                compilation.cache_key,
                compilation.graph.to_json(),
                compilation.timeline.to_json(),
                compilation.options.to_json(),
                compilation.decision.to_json(),
                compilation.physical.plan_epoch,
                [worker_spec.to_wire()],
            )
            extension.tllm_commit_workload_decision(
                compilation.decision.signature,
                compilation.physical.plan_epoch,
            )

            # First-epoch commit only records the vote. Hook topology is an
            # activation side effect and cannot leak before global quorum.
            self.assertFalse(hasattr(model, residual_runtime.MODEL_HOOK_FLAG))
            self.assertIs(layer.forward.__func__, original_forward_function)
            self.assertIs(state.decode_row_idx, previous_decode_rows)
            self.assertFalse(hasattr(runner.sampler, "_tllm_runtime"))

            extension.tllm_abort_workload_decision(
                compilation.decision.signature,
                compilation.physical.plan_epoch,
            )

        self.assertFalse(hasattr(model, residual_runtime.MODEL_HOOK_FLAG))
        self.assertEqual(layer.forward, original_forward)
        self.assertEqual(model.compute_logits, original_compute_logits)
        self.assertIs(state.decode_row_idx, previous_decode_rows)
        self.assertEqual(state.tap_handles, {})
        self.assertFalse(hasattr(runner.sampler, "_tllm_runtime"))
        self.assertIs(state.workload_compilation, previous_compilation)
        self.assertIs(state.workload_execution, previous_execution)
        self.assertIs(state.physical_executor, previous_execution)

    def test_parent_aborts_all_staged_workers_after_partial_commit_failure(self) -> None:
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.graph_policy import GraphMode
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.state import ResidualRuntimeState
        from tllm.workgraph.anchors import BackendResolution

        spec = WorkloadWorkerSpec(
            f"{__name__}:_reconstruct_native_mp_workload", {}
        )
        runtime = TllmRuntime(
            workloads=(_NativeMPWorkload(),),
            workload_memory_budget_bytes=64,
            workload_worker_specs=(spec,),
        )
        llm = _FakeLLM()
        second = TllmWorkerExtension()
        llm.llm_engine.engine_core.handlers.append(second)
        llm.llm_engine.engine_core.single_timeline_reply = True
        for handler in llm.llm_engine.engine_core.handlers:
            handler.model_runner = SimpleNamespace(device="cpu")
        llm.llm_engine.engine_core.fail_method = "tllm_commit_workload_decision"
        llm.llm_engine.engine_core.fail_handler_index = 1
        plane = ControlPlane(llm)
        state = ResidualRuntimeState()
        previous_compilation = object()
        previous_execution = object()
        state.workload_compilation = previous_compilation
        state.workload_execution = previous_execution  # type: ignore[assignment]
        state.physical_executor = previous_execution  # type: ignore[assignment]
        state.workgraph_backend_resolution = BackendResolution(
            capabilities=V027MRv2Driver().capabilities(), graph_mode=GraphMode.EAGER
        )

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            with self.assertRaisesRegex(RuntimeError, "simulated worker 1 failure"):
                runtime._coordinate_worker_compilation(plane)

        methods = [call[0] for call in llm.llm_engine.engine_core.calls]
        self.assertEqual(
            methods,
            [
                "tllm_workload_timeline",
                "tllm_prepare_workload_decision",
                "tllm_commit_workload_decision",
                "tllm_abort_workload_decision",
            ],
        )
        self.assertIs(state.workload_compilation, previous_compilation)
        self.assertIs(state.workload_execution, previous_execution)
        self.assertIs(state.physical_executor, previous_execution)
        self.assertIsNone(getattr(llm.llm_engine.engine_core.handler, "_tllm_staged_workload", None))
        self.assertIsNone(getattr(second, "_tllm_staged_workload", None))

    def test_parent_fail_stops_after_partial_activation_failure(self) -> None:
        """An activation prefix is never rolled back rank-by-rank."""

        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.graph_policy import GraphMode
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.state import ResidualRuntimeState
        from tllm.workgraph.anchors import BackendResolution

        spec = WorkloadWorkerSpec(
            f"{__name__}:_reconstruct_native_mp_workload", {}
        )
        runtime = TllmRuntime(
            workloads=(_NativeMPWorkload(),),
            workload_memory_budget_bytes=64,
            workload_worker_specs=(spec,),
        )
        llm = _FakeLLM()
        second = TllmWorkerExtension()
        llm.llm_engine.engine_core.handlers.append(second)
        llm.llm_engine.engine_core.single_timeline_reply = True
        for handler in llm.llm_engine.engine_core.handlers:
            handler.model_runner = SimpleNamespace(device="cpu")
        llm.llm_engine.engine_core.fail_method = "tllm_activate_workload_decision"
        llm.llm_engine.engine_core.fail_handler_index = 1
        plane = ControlPlane(llm)
        state = ResidualRuntimeState(
            workgraph_backend_resolution=BackendResolution(
                capabilities=V027MRv2Driver().capabilities(),
                graph_mode=GraphMode.EAGER,
            )
        )

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            with self.assertRaisesRegex(RuntimeError, "fail-stop"):
                runtime._coordinate_worker_compilation(plane)

        self.assertEqual(
            [call[0] for call in llm.llm_engine.engine_core.calls],
            [
                "tllm_workload_timeline",
                "tllm_prepare_workload_decision",
                "tllm_commit_workload_decision",
                "tllm_activate_workload_decision",
            ],
        )
        first_staged = llm.llm_engine.engine_core.handler._tllm_staged_workload
        second_staged = second._tllm_staged_workload
        self.assertIsNotNone(first_staged)
        self.assertIsNotNone(second_staged)
        assert first_staged is not None and second_staged is not None
        self.assertTrue(first_staged.activated)
        self.assertFalse(second_staged.activated)
        calls_before_retry = len(llm.llm_engine.engine_core.calls)
        with self.assertRaisesRegex(RuntimeError, "fail-stopped"):
            runtime._coordinate_worker_compilation(plane)
        self.assertEqual(len(llm.llm_engine.engine_core.calls), calls_before_retry)

    def test_truncated_finalize_quorum_fail_stops_runtime(self) -> None:
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.graph_policy import GraphMode
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.state import ResidualRuntimeState
        from tllm.workgraph.anchors import BackendResolution

        spec = WorkloadWorkerSpec(
            f"{__name__}:_reconstruct_native_mp_workload", {}
        )
        runtime = TllmRuntime(
            workloads=(_NativeMPWorkload(),),
            workload_memory_budget_bytes=64,
            workload_worker_specs=(spec,),
        )
        llm = _FakeLLM()
        second = TllmWorkerExtension()
        llm.llm_engine.engine_core.handlers.append(second)
        llm.llm_engine.engine_core.single_timeline_reply = True
        llm.llm_engine.engine_core.reply_limits[
            "tllm_finalize_workload_decision"
        ] = 1
        for handler in llm.llm_engine.engine_core.handlers:
            handler.model_runner = SimpleNamespace(device="cpu")
        plane = ControlPlane(llm)
        state = ResidualRuntimeState(
            workgraph_backend_resolution=BackendResolution(
                capabilities=V027MRv2Driver().capabilities(),
                graph_mode=GraphMode.EAGER,
            )
        )

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            with self.assertRaisesRegex(RuntimeError, "runtime is fail-stopped"):
                runtime._coordinate_worker_compilation(plane)
        self.assertNotIn(
            "tllm_abort_workload_decision",
            [call[0] for call in llm.llm_engine.engine_core.calls],
        )

    def test_activation_rpc_rejects_non_json_identity_scalars(self) -> None:
        extension = TllmWorkerExtension()
        with self.assertRaisesRegex(TypeError, "SHA-256"):
            extension.tllm_activate_workload_decision("not-a-signature", 1)
        with self.assertRaisesRegex(TypeError, "positive integer"):
            extension.tllm_activate_workload_decision("a" * 64, True)

    def test_one_shot_install_preserves_post_activation_failure(self) -> None:
        extension = TllmWorkerExtension()
        signature = "a" * 64
        with (
            mock.patch.object(
                extension,
                "tllm_prepare_workload_decision",
                return_value={"signature": signature},
            ),
            mock.patch.object(extension, "tllm_commit_workload_decision"),
            mock.patch.object(extension, "tllm_activate_workload_decision"),
            mock.patch.object(
                extension,
                "tllm_finalize_workload_decision",
                side_effect=RuntimeError("finalize failed"),
            ),
            mock.patch.object(extension, "tllm_abort_workload_decision") as abort,
        ):
            with self.assertRaisesRegex(RuntimeError, "finalize failed"):
                extension.tllm_install_workload_decision(
                    "b" * 64, "{}", "{}", "{}", "{}", 1
                )
        abort.assert_not_called()

    def test_parent_aborts_staged_prefix_after_prepare_rpc_failure(self) -> None:
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.graph_policy import GraphMode
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.state import ResidualRuntimeState
        from tllm.workgraph.anchors import BackendResolution

        spec = WorkloadWorkerSpec(
            f"{__name__}:_reconstruct_native_mp_workload", {}
        )
        runtime = TllmRuntime(
            workloads=(_NativeMPWorkload(),),
            workload_memory_budget_bytes=64,
            workload_worker_specs=(spec,),
        )
        llm = _FakeLLM()
        second = TllmWorkerExtension()
        llm.llm_engine.engine_core.handlers.append(second)
        llm.llm_engine.engine_core.single_timeline_reply = True
        for handler in llm.llm_engine.engine_core.handlers:
            handler.model_runner = SimpleNamespace(device="cpu")
        llm.llm_engine.engine_core.fail_method = "tllm_prepare_workload_decision"
        llm.llm_engine.engine_core.fail_handler_index = 1
        plane = ControlPlane(llm)
        state = ResidualRuntimeState(
            workgraph_backend_resolution=BackendResolution(
                capabilities=V027MRv2Driver().capabilities(),
                graph_mode=GraphMode.EAGER,
            )
        )

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            with self.assertRaisesRegex(RuntimeError, "simulated worker 1 failure"):
                runtime._coordinate_worker_compilation(plane)

        self.assertEqual(
            [call[0] for call in llm.llm_engine.engine_core.calls],
            [
                "tllm_workload_timeline",
                "tllm_prepare_workload_decision",
                "tllm_abort_workload_decision",
            ],
        )
        self.assertIsNone(llm.llm_engine.engine_core.handler._tllm_staged_workload)
        self.assertIsNone(second._tllm_staged_workload)
        self.assertIsNone(state.workload_compilation)
        self.assertIsNone(state.workload_execution)

    def test_finalize_failure_fail_stops_committed_epoch(self) -> None:
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.graph_policy import GraphMode
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.state import ResidualRuntimeState
        from tllm.workgraph.anchors import BackendResolution

        spec = WorkloadWorkerSpec(
            f"{__name__}:_reconstruct_native_mp_workload", {}
        )
        runtime = TllmRuntime(
            workloads=(_NativeMPWorkload(),),
            workload_memory_budget_bytes=64,
            workload_worker_specs=(spec,),
        )
        llm = _FakeLLM()
        second = TllmWorkerExtension()
        llm.llm_engine.engine_core.handlers.append(second)
        llm.llm_engine.engine_core.single_timeline_reply = True
        for handler in llm.llm_engine.engine_core.handlers:
            handler.model_runner = SimpleNamespace(device="cpu")
        llm.llm_engine.engine_core.fail_method = "tllm_finalize_workload_decision"
        llm.llm_engine.engine_core.fail_handler_index = 1
        plane = ControlPlane(llm)
        state = ResidualRuntimeState(
            workgraph_backend_resolution=BackendResolution(
                capabilities=V027MRv2Driver().capabilities(),
                graph_mode=GraphMode.EAGER,
            )
        )

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            with self.assertRaisesRegex(RuntimeError, "runtime is fail-stopped"):
                runtime._coordinate_worker_compilation(plane)
            compilation = runtime.workload_compilation
            self.assertIsNotNone(compilation)
            self.assertIsNotNone(state.workload_execution)
            assert compilation is not None
            self.assertEqual(
                state.workload_compilation.decision.signature,  # type: ignore[union-attr]
                compilation.decision.signature,
            )
            self.assertNotIn(
                "tllm_abort_workload_decision",
                [call[0] for call in llm.llm_engine.engine_core.calls],
            )
            calls_before_retry = len(llm.llm_engine.engine_core.calls)
            with self.assertRaisesRegex(RuntimeError, "fail-stopped"):
                runtime._coordinate_worker_compilation(plane)
            self.assertEqual(len(llm.llm_engine.engine_core.calls), calls_before_retry)

    def test_epoch_drain_requires_safe_point_before_finalize(self) -> None:
        """A committed replacement cannot close its prior epoch early."""

        from tllm.backends.timeline import export_main_workload_timeline
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.state import ResidualRuntimeState
        from tllm.runtime.workload_execution import (
            NativeWorkloadExecution,
            NativeWorkloadExecutionStore,
        )

        class _Executor:
            def prepare_step(self, _step_id: int) -> None:
                return None

            def finish_step(self, _step_id: int) -> None:
                return None

        runtime = TllmRuntime(
            workloads=(_NativeMPWorkload(),), workload_memory_budget_bytes=64
        )
        compilation = runtime.compile_workloads(
            export_main_workload_timeline(
                V027MRv2Driver().capabilities(),
                graph_mode="eager",
                static_headroom_bytes=64,
            )
        )
        previous = NativeWorkloadExecution(
            compilation=SimpleNamespace(
                physical=SimpleNamespace(plan_epoch=0)
            ),
            executor=_Executor(),  # type: ignore[arg-type]
            completion_anchor="done",
            _owned_consumers=(),
        )
        state = ResidualRuntimeState(
            workload_compilation=previous.compilation,  # type: ignore[arg-type]
            workload_execution=previous,
            workload_execution_store=NativeWorkloadExecutionStore(previous),
            physical_executor=previous,
        )
        extension = TllmWorkerExtension()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            extension.tllm_prepare_workload_decision(
                compilation.cache_key,
                compilation.graph.to_json(),
                compilation.timeline.to_json(),
                compilation.options.to_json(),
                compilation.decision.to_json(),
                compilation.physical.plan_epoch,
            )
            staged = extension._tllm_staged_workload
            assert staged is not None
            staged.execution = NativeWorkloadExecution(
                compilation=compilation,
                executor=_Executor(),  # type: ignore[arg-type]
                completion_anchor="done",
                _owned_consumers=(),
            )
            staged.bindings = SimpleNamespace(  # type: ignore[assignment]
                residual_paths={
                    ResidualLocator(0, "block_output", "decode"): "model.layers.0"
                }
            )
            with self.assertRaisesRegex(RuntimeError, "hook topology"):
                extension.tllm_commit_workload_decision(
                    compilation.decision.signature, compilation.physical.plan_epoch
                )
            self.assertIs(state.workload_execution, previous)
            # Bindings alone are not a topology change.  Identical residual
            # paths and hidden-tap requirements may safely stage a new epoch.
            staged.bindings = SimpleNamespace(residual_paths={})  # type: ignore[assignment]
            committed = extension.tllm_commit_workload_decision(
                compilation.decision.signature, compilation.physical.plan_epoch
            )
            self.assertTrue(committed["committed"])
            self.assertIs(state.workload_execution, previous)
            store = state.workload_execution_store
            assert store is not None
            self.assertFalse(store.staged_activated)

            extension.tllm_activate_workload_decision(
                compilation.decision.signature, compilation.physical.plan_epoch
            )
            self.assertTrue(store.staged_activated)

            blocked = extension.tllm_finalize_workload_decision(
                compilation.decision.signature, compilation.physical.plan_epoch
            )
            self.assertEqual(blocked["finalized"], False)
            self.assertEqual(blocked["drained"], False)

            store.prepare_step(10)
            store.finish_step(10)
            store.prepare_step(11)
            self.assertEqual(store.retired_count, 1)

            with self.assertRaisesRegex(RuntimeError, "after global activation"):
                extension.tllm_abort_workload_decision(
                    compilation.decision.signature,
                    compilation.physical.plan_epoch,
                )

            drained = extension.tllm_drain_workload_epochs(
                compilation.decision.signature, compilation.physical.plan_epoch
            )
            self.assertEqual(drained["drained"], True)
            self.assertEqual(drained["closed_epochs"], 1)
            finalized = extension.tllm_finalize_workload_decision(
                compilation.decision.signature, compilation.physical.plan_epoch
            )
            self.assertEqual(finalized["finalized"], True)
            self.assertEqual(finalized["drained"], True)
            self.assertEqual(
                extension.tllm_finalize_workload_decision(
                    compilation.decision.signature, compilation.physical.plan_epoch
                )["finalized"],
                True,
            )
            self.assertEqual(
                extension.tllm_drain_workload_epochs(
                    compilation.decision.signature, compilation.physical.plan_epoch
                )["drained"],
                True,
            )
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                extension.tllm_drain_workload_epochs(
                    "b" * 64, compilation.physical.plan_epoch
                )

    def test_epoch_drain_rpc_uses_only_scalar_json_arguments(self) -> None:
        llm = _FakeLLM()
        plane = ControlPlane(llm)
        extension = llm.llm_engine.engine_core.handler
        signature = "a" * 64
        extension._tllm_last_finalized_workload = (signature, 3)

        (reply,) = plane.drain_workload_epochs(signature, 3)

        self.assertTrue(reply["drained"])
        method, args, _timeout = llm.llm_engine.engine_core.calls[-1]
        self.assertEqual(method, "tllm_drain_workload_epochs")
        self.assertEqual(args, (signature, 3))

    def test_shape_specialization_peek_is_typed_and_non_destructive(self) -> None:
        signature = "c" * 64
        state, execution = _shape_specialization_state(signature, 7)
        request = _shape_specialization_request(signature, 7)
        self.assertTrue(execution._shape_specializations.enqueue(request))
        llm = _FakeLLM()
        plane = ControlPlane(llm)

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            (first,) = plane.peek_shape_specialization_requests()
            (second,) = plane.peek_shape_specialization_requests()

        self.assertEqual(first, second)
        self.assertEqual(first.signature, signature)
        self.assertEqual(first.plan_epoch, 7)
        self.assertEqual(first.requests, (request,))
        self.assertEqual(
            [call[0] for call in llm.llm_engine.engine_core.calls],
            [
                "tllm_peek_shape_specialization_requests",
                "tllm_peek_shape_specialization_requests",
            ],
        )
        self.assertEqual(execution.peek_shape_specialization_requests(), (request,))

    def test_shape_specialization_ack_is_exact_and_preserves_on_failure(self) -> None:
        signature = "c" * 64
        state, execution = _shape_specialization_state(signature, 7)
        request = _shape_specialization_request(signature, 7)
        self.assertTrue(execution._shape_specializations.enqueue(request))
        llm = _FakeLLM()
        plane = ControlPlane(llm)
        extension = llm.llm_engine.engine_core.handler

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                extension.tllm_ack_shape_specialization_requests(
                    "d" * 64, 7, [request.dedup_key]
                )
            with self.assertRaisesRegex(ValueError, "no matching pending"):
                extension.tllm_ack_shape_specialization_requests(
                    signature, 7, ["d" * 64]
                )
            self.assertEqual(execution.peek_shape_specialization_requests(), (request,))

            (acknowledged,) = plane.ack_shape_specialization_requests(
                signature, 7, [request.dedup_key]
            )

        self.assertEqual(acknowledged, (request.dedup_key,))
        self.assertEqual(execution.peek_shape_specialization_requests(), ())
        method, args, _timeout = llm.llm_engine.engine_core.calls[-1]
        self.assertEqual(method, "tllm_ack_shape_specialization_requests")
        self.assertEqual(args, (signature, 7, [request.dedup_key]))

    def test_shape_specialization_reply_rejects_unknown_fields_and_identity_mix(self) -> None:
        signature = "c" * 64
        request = _shape_specialization_request(signature, 7)
        payload = {
            "signature": signature,
            "plan_epoch": 7,
            "requests": [request.to_wire()],
        }
        with self.assertRaisesRegex(ValueError, "invalid fields"):
            WorkerShapeSpecializationReply.from_wire(
                {**payload, "unexpected": True}
            )

        mixed = request.to_wire()
        mixed["base_signature"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "identity does not match"):
            WorkerShapeSpecializationReply.from_wire(
                {**payload, "requests": [mixed]}
            )

        malformed = request.to_wire()
        malformed["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "invalid fields"):
            WorkerShapeSpecializationReply.from_wire(
                {**payload, "requests": [malformed]}
            )

    def test_parent_installs_and_acknowledges_worker_shape_miss_exactly(self) -> None:
        """The parent consumes a TP=1 miss only after replacement install."""

        from tllm.runtime.runtime import TllmRuntime

        signature = "c" * 64
        request = _shape_specialization_request(signature, 7)
        current = SimpleNamespace(
            decision=SimpleNamespace(signature=signature),
            physical=SimpleNamespace(plan_epoch=7),
        )
        replacement = SimpleNamespace(
            decision=SimpleNamespace(signature="d" * 64),
            physical=SimpleNamespace(plan_epoch=8),
        )
        runtime = TllmRuntime()
        runtime._workload_compilation = current  # type: ignore[assignment]
        plane = mock.Mock()
        plane.peek_shape_specialization_requests.return_value = [
            WorkerShapeSpecializationReply(signature, 7, (request,))
        ]
        plane.ack_shape_specialization_requests.return_value = [(request.dedup_key,)]

        def compile_replacement(
            requests: object, *, publish: bool = True
        ) -> bool:
            self.assertEqual(requests, (request,))
            self.assertFalse(publish)
            runtime._workload_compilation = replacement  # type: ignore[assignment]
            return True

        with (
            mock.patch.object(
                runtime,
                "apply_region_shape_specializations",
                side_effect=compile_replacement,
            ) as compile_shapes,
            mock.patch.object(
                runtime,
                "_install_workload_compilation_on_workers",
            ) as install,
        ):
            self.assertTrue(runtime.apply_pending_worker_shape_specializations(plane))

        compile_shapes.assert_called_once_with((request,), publish=False)
        install.assert_called_once_with(plane, replacement)
        plane.ack_shape_specialization_requests.assert_called_once_with(
            signature, 7, (request.dedup_key,)
        )
        self.assertIs(runtime.workload_compilation, replacement)

    def test_parent_shape_miss_install_failure_restores_parent_and_does_not_ack(self) -> None:
        from tllm.runtime.runtime import TllmRuntime

        signature = "c" * 64
        request = _shape_specialization_request(signature, 7)
        current = SimpleNamespace(
            decision=SimpleNamespace(signature=signature),
            physical=SimpleNamespace(plan_epoch=7),
        )
        replacement = SimpleNamespace(
            decision=SimpleNamespace(signature="d" * 64),
            physical=SimpleNamespace(plan_epoch=8),
        )
        runtime = TllmRuntime()
        runtime._workload_compilation = current  # type: ignore[assignment]
        previous_options = runtime._workload_compile_options
        previous_compiler = runtime._workload_compiler
        plane = mock.Mock()
        plane.peek_shape_specialization_requests.return_value = [
            WorkerShapeSpecializationReply(signature, 7, (request,))
        ]

        def compile_replacement(
            _requests: object, *, publish: bool = True
        ) -> bool:
            self.assertFalse(publish)
            runtime._workload_compile_options = SimpleNamespace()
            runtime._workload_compiler = SimpleNamespace()
            runtime._workload_compilation = replacement  # type: ignore[assignment]
            return True

        with (
            mock.patch.object(
                runtime,
                "apply_region_shape_specializations",
                side_effect=compile_replacement,
            ),
            mock.patch.object(
                runtime,
                "_install_workload_compilation_on_workers",
                side_effect=RuntimeError("worker install failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "worker install failed"):
                runtime.apply_pending_worker_shape_specializations(plane)

        self.assertIs(runtime.workload_compilation, current)
        self.assertIs(runtime._workload_compile_options, previous_options)
        self.assertIs(runtime._workload_compiler, previous_compiler)
        plane.ack_shape_specialization_requests.assert_not_called()

    def test_parent_shape_miss_rejects_non_tp1_or_stale_worker_envelopes(self) -> None:
        from tllm.runtime.runtime import TllmRuntime

        signature = "c" * 64
        request = _shape_specialization_request(signature, 7)
        current = SimpleNamespace(
            decision=SimpleNamespace(signature=signature),
            physical=SimpleNamespace(plan_epoch=7),
        )
        runtime = TllmRuntime()
        runtime._workload_compilation = current  # type: ignore[assignment]
        plane = mock.Mock()
        reply = WorkerShapeSpecializationReply(signature, 7, (request,))
        plane.peek_shape_specialization_requests.return_value = [reply, reply]

        with self.assertRaisesRegex(RuntimeError, "exactly one TP=1"):
            runtime.apply_pending_worker_shape_specializations(plane)

        plane.peek_shape_specialization_requests.return_value = [
            WorkerShapeSpecializationReply("d" * 64, 7, ())
        ]
        self.assertFalse(runtime.apply_pending_worker_shape_specializations(plane))

        stale_request = _shape_specialization_request("d" * 64, 7)
        plane.peek_shape_specialization_requests.return_value = [
            WorkerShapeSpecializationReply("d" * 64, 7, (stale_request,))
        ]
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            runtime.apply_pending_worker_shape_specializations(plane)

    def test_next_generate_polls_parent_shape_miss_loop_without_retrying_step(self) -> None:
        from tllm.runtime.runtime import TllmRuntime

        runtime = TllmRuntime(workloads=(_NativeMPWorkload(),))
        plane = object()
        events: list[str] = []
        llm = SimpleNamespace(
            generate=lambda prompt: events.append(f"generate:{prompt}") or "output"
        )
        with mock.patch.object(
            runtime,
            "apply_pending_worker_shape_specializations",
            side_effect=lambda seen_plane: events.append(
                "poll" if seen_plane is plane else "wrong-plane"
            )
            or False,
        ):
            runtime._install_shape_specialization_poll(llm, plane)
            runtime._install_shape_specialization_poll(llm, plane)
            self.assertEqual(llm.generate("next"), "output")

        self.assertEqual(events, ["poll", "generate:next"])

    def test_shape_generate_wrapper_rejects_fail_stopped_runtime(self) -> None:
        from tllm.runtime.runtime import TllmRuntime

        runtime = TllmRuntime(workloads=(_NativeMPWorkload(),))
        original_generate = mock.Mock(return_value="unexpected")
        llm = SimpleNamespace(generate=original_generate)
        runtime._install_shape_specialization_poll(llm, object())
        runtime._fail_stop_workloads("activation quorum lost")

        with self.assertRaisesRegex(RuntimeError, "fail-stopped"):
            llm.generate("next")
        original_generate.assert_not_called()

    def test_dummy_native_worker_reconstructs_without_dispatch_consumer(self) -> None:
        from tllm.backends.timeline import export_main_workload_timeline
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.specialization import RuntimeSpecializationSpec
        from tllm.runtime.state import ResidualRuntimeState

        workload = DummyWorkload(DummyConfig(enable_async=False))
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
        )
        compilation = runtime.compile_workloads(
            export_main_workload_timeline(
                V027MRv2Driver().capabilities(),
                graph_mode="eager",
                static_headroom_bytes=1 << 20,
            )
        )
        worker_spec = workload.native_workload_resources().worker_spec
        self.assertIsNotNone(worker_spec)
        assert worker_spec is not None
        extension = TllmWorkerExtension()
        extension.model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            max_num_reqs=2,
            max_num_tokens=16,
            model_config=SimpleNamespace(
                dtype=torch.float32,
                get_hidden_size=lambda: 4,
            ),
            model=_HookModel(),
        )
        state = ResidualRuntimeState()

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch(
                "tllm.runtime.vllm_patch.port_runtime_hooks."
                "setup_runtime_hooks_if_active"
            ) as setup,
        ):
            report = extension.tllm_install_workload_decision(
                compilation.cache_key,
                compilation.graph.to_json(),
                compilation.timeline.to_json(),
                compilation.options.to_json(),
                compilation.decision.to_json(),
                compilation.physical.plan_epoch,
                [worker_spec.to_wire()],
            )

        self.assertTrue(report["native"])
        self.assertEqual(state.consumer_registry, None)
        self.assertFalse(state.native_workloads_pending)
        self.assertEqual(
            state.residual_raw_paths,
            {ResidualLocator(0, "block_output", "decode"): "model.layers.0"},
        )
        setup.assert_called_once_with(core=residual_runtime, runner=extension.model_runner)
        execution = state.workload_execution
        self.assertIsNotNone(execution)
        assert execution is not None
        self.assertEqual(
            execution.artifact_report,
            tuple(
                (region.region_id, region.artifact_key)
                for region in compilation.decision.execution_regions
            ),
        )
        execution.close()

    def test_workload_status_round_trips_namespaced_worker_diagnostics(self) -> None:
        from tllm.backends.timeline import export_main_workload_timeline
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.specialization import RuntimeSpecializationSpec
        from tllm.runtime.state import ResidualRuntimeState

        workload = DummyWorkload(DummyConfig(enable_async=False))
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
        )
        compilation = runtime.compile_workloads(
            export_main_workload_timeline(
                V027MRv2Driver().capabilities(),
                graph_mode="eager",
                static_headroom_bytes=1 << 20,
            )
        )
        worker_spec = workload.native_workload_resources().worker_spec
        assert worker_spec is not None
        extension = TllmWorkerExtension()
        extension.model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            max_num_reqs=2,
            max_num_tokens=16,
            model_config=SimpleNamespace(
                dtype=torch.float32,
                get_hidden_size=lambda: 4,
            ),
            model=_HookModel(),
        )
        state = ResidualRuntimeState()
        llm = _FakeLLM()
        llm.llm_engine.engine_core.handler = extension
        llm.llm_engine.engine_core.handlers = [extension]

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch(
                "tllm.runtime.vllm_patch.port_runtime_hooks."
                "setup_runtime_hooks_if_active"
            ),
        ):
            extension.tllm_install_workload_decision(
                compilation.cache_key,
                compilation.graph.to_json(),
                compilation.timeline.to_json(),
                compilation.options.to_json(),
                compilation.decision.to_json(),
                compilation.physical.plan_epoch,
                [worker_spec.to_wire()],
            )
            statuses = ControlPlane(llm).workload_status()

        self.assertEqual(len(statuses), 1)
        self.assertEqual(
            statuses[0]["diagnostics"],
            {
                "dummy": {
                    "consumed_batches": 0.0,
                    "consumed_rows": 0.0,
                    "dropped_batches": 0.0,
                    "last_mean_abs": 0.0,
                    "last_noise_std": 0.0,
                    "pending": 0.0,
                    "processed_batches": 0.0,
                    "processed_rows": 0.0,
                }
            },
        )
        assert state.workload_execution is not None
        state.workload_execution.close()

    def test_workload_status_reports_current_epoch_until_safe_point_promotion(self) -> None:
        from tllm.runtime.state import ResidualRuntimeState
        from tllm.runtime.workload_execution import (
            NativeWorkloadExecution,
            NativeWorkloadExecutionStore,
        )

        def compilation(signature: str, epoch: int) -> SimpleNamespace:
            return SimpleNamespace(
                decision=SimpleNamespace(signature=signature),
                physical=SimpleNamespace(plan_epoch=epoch),
                graph=SimpleNamespace(),
            )

        old_compilation = compilation("a" * 64, 1)
        replacement_compilation = compilation("b" * 64, 2)
        old_execution = NativeWorkloadExecution(
            compilation=old_compilation,  # type: ignore[arg-type]
            executor=mock.Mock(),  # type: ignore[arg-type]
            completion_anchor="step.end",
            _owned_consumers=(),
            _artifact_report=(("old-region", "c" * 64),),
            _diagnostics_readers={"old": lambda: {"observed": 1}},
        )
        replacement = NativeWorkloadExecution(
            compilation=replacement_compilation,  # type: ignore[arg-type]
            executor=mock.Mock(),  # type: ignore[arg-type]
            completion_anchor="step.end",
            _owned_consumers=(),
            _artifact_report=(("replacement-region", "d" * 64),),
            _diagnostics_readers={"replacement": lambda: {"observed": 2}},
        )
        store = NativeWorkloadExecutionStore(old_execution)
        store.stage(replacement, activation_required=True)
        self.assertTrue(store.activate_staged(2))
        state = ResidualRuntimeState(
            workload_compilation=old_compilation,  # type: ignore[arg-type]
            workload_execution=old_execution,
            workload_execution_store=store,
        )
        extension = TllmWorkerExtension()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            before = extension.tllm_workload_status()
            self.assertEqual(before["signature"], "a" * 64)
            self.assertEqual(before["plan_epoch"], 1)
            self.assertEqual(before["artifacts"], [
                {"region_id": "old-region", "artifact_key": "c" * 64}
            ])
            self.assertEqual(before["diagnostics"], {"old": {"observed": 1}})

            residual_runtime.prepare_physical_workgraph_step(7)
            after = extension.tllm_workload_status()

        self.assertIs(state.workload_execution, replacement)
        self.assertEqual(after["signature"], "b" * 64)
        self.assertEqual(after["plan_epoch"], 2)
        self.assertEqual(after["artifacts"], [
            {"region_id": "replacement-region", "artifact_key": "d" * 64}
        ])
        self.assertEqual(after["diagnostics"], {"replacement": {"observed": 2}})
        store.close()

    def test_workload_status_reconstructs_multiple_diagnostic_namespaces(self) -> None:
        from tllm.backends.timeline import export_main_workload_timeline
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.specialization import RuntimeSpecializationSpec
        from tllm.runtime.state import ResidualRuntimeState

        alpha = _StatusAlphaWorkload(101)
        beta = _StatusBetaWorkload(202)
        runtime = TllmRuntime(
            workloads=(alpha, beta), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
        )
        compilation = runtime.compile_workloads(
            export_main_workload_timeline(
                V027MRv2Driver().capabilities(),
                graph_mode="eager",
                static_headroom_bytes=1 << 20,
            )
        )
        specs = tuple(
            workload.native_workload_resources().worker_spec
            for workload in (alpha, beta)
        )
        self.assertTrue(all(spec is not None for spec in specs))
        extension = TllmWorkerExtension()
        extension.model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            max_num_reqs=2,
            max_num_tokens=16,
            model_config=SimpleNamespace(
                dtype=torch.float32,
                get_hidden_size=lambda: 4,
            ),
        )
        state = ResidualRuntimeState()
        llm = _FakeLLM()
        llm.llm_engine.engine_core.handler = extension
        llm.llm_engine.engine_core.handlers = [extension]

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            extension.tllm_install_workload_decision(
                compilation.cache_key,
                compilation.graph.to_json(),
                compilation.timeline.to_json(),
                compilation.options.to_json(),
                compilation.decision.to_json(),
                compilation.physical.plan_epoch,
                [spec.to_wire() for spec in specs if spec is not None],
            )
            statuses = ControlPlane(llm).workload_status()

        self.assertEqual(len(statuses), 1)
        self.assertEqual(
            statuses[0]["diagnostics"],
            {
                "status.alpha": {"metric": 101},
                "status.beta": {"metric": 202},
            },
        )
        assert state.workload_execution is not None
        state.workload_execution.close()

    def test_workload_status_rejects_non_json_diagnostics(self) -> None:
        llm = _FakeLLM()
        llm.llm_engine.engine_core.handler.tllm_workload_status = lambda: {
            "pid": os.getpid(),
            "signature": "a" * 64,
            "plan_epoch": 1,
            "artifacts": [],
            "state_versions": [],
            "diagnostics": {"dummy": {"count": float("nan")}},
        }
        with self.assertRaisesRegex(ValueError, "finite"):
            ControlPlane(llm).workload_status()

    def test_failed_native_install_clears_pending_state_and_partial_bindings(self) -> None:
        from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
        from tllm.backends.timeline import export_main_workload_timeline
        from tllm.consumers.esamp.region_workload import (
            ESampConfig,
            ESampWorkload,
        )
        from tllm.runtime.runtime import TllmRuntime
        from tllm.runtime.specialization import RuntimeSpecializationSpec
        from tllm.runtime.state import ResidualRuntimeState

        workload = ESampWorkload(ESampConfig())
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        specialization = RuntimeSpecializationSpec(
            2, 4, torch.float32, torch.device("cpu")
        )
        runtime.cold_specialize_workloads(specialization)
        compilation = runtime.compile_workloads(
            export_main_workload_timeline(
                V027MRv2Driver().capabilities(),
                graph_mode="eager",
                static_headroom_bytes=1 << 20,
            )
        )
        worker_spec = workload.native_workload_resources().worker_spec
        assert worker_spec is not None
        extension = TllmWorkerExtension()
        extension.model_runner = SimpleNamespace(
            device=torch.device("cpu"),
            max_num_reqs=2,
            max_num_tokens=16,
            model_config=SimpleNamespace(
                dtype=torch.float32,
                get_hidden_size=lambda: 4,
            ),
            model=_HookModel(),
        )
        state = ResidualRuntimeState()

        with mock.patch.object(residual_runtime, "RUNTIME", state), mock.patch(
            "tllm.runtime.vllm_patch.port_runtime_hooks."
            "setup_runtime_hooks_if_active",
            side_effect=RuntimeError("tap install failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "tap install failed"):
                extension.tllm_install_workload_decision(
                    compilation.cache_key,
                    compilation.graph.to_json(),
                    compilation.timeline.to_json(),
                    compilation.options.to_json(),
                    compilation.decision.to_json(),
                    compilation.physical.plan_epoch,
                    [worker_spec.to_wire()],
                )

        self.assertFalse(state.native_workloads_pending)
        self.assertIsNone(state.workload_execution)
        self.assertIsNone(state.workload_compilation)
        self.assertEqual(state.residual_raw_paths, {})

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

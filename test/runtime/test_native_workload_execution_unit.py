from __future__ import annotations

import contextlib
import unittest
from unittest import mock
from collections.abc import Hashable, Mapping
from dataclasses import replace
from threading import Event, Thread
from typing import Any, cast

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.consumers.dummy.workload import DummyConfig, DummyWorkload
from tllm.contracts.hidden_locator import HiddenLocatorCapability, HiddenPhase, HiddenSite
from tllm.contracts.consumer_stats import TrainingStats
from tllm.contracts.shape_specialization import RegionShapeSpecializationRequest
from tllm.contracts.runtime_context import CompilationConfigLike
from tllm.contracts.workload_ports import SKIPPED
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime import residual_runtime
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_execution import (
    ExecutionLifecycleBinding,
    NativeWorkloadExecution,
    NoExecutionLifecycle,
    _RegionCudaGraph,
    _NativeShapeSpecializationQueue,
    _RegionValueStore,
    _bind_physical_region_nodes,
    lower_signed_materialization_bindings,
    _region_shape_specialization_request,
    StateVersionBinding,
    StateVersionRegistry,
    WorkloadExecutionBindings,
)
from tllm.runtime.workload_execution import _hidden_value
from tllm.runtime.workload_plugins import reconstruct_worker_bindings
from tllm.runtime.workload_bindings import (
    LocalRuntimeServices,
    PerRequestStats,
    RuntimeInputBinding,
    RuntimeOutputBinding,
    RuntimePortRegistry,
    WorkerRuntimeBinding,
)
from tllm.runtime.vllm_patch import common_hooks
from tllm.runtime.scheduling.workgraph_executor import (
    PhysicalExecutionContext,
    PhysicalExecutionKind,
    ShapeSpecializationMiss,
)
from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.physical import (
    CapturePolicy,
    FallbackPolicy,
    PhysicalNodePlan,
    StreamClass,
)
from tllm.workload.region_compiler import (
    RegionCallableArtifact,
    materialize_region_artifacts,
)
from tllm.schedule import (
    BackendObservableEvent,
    CompileOptions,
    HiddenLayerKind,
    MainWorkloadTimeline,
    RegionInputBinding,
    RegionInputKind,
    RegionPort,
    ResidualResourceBinding,
    ScheduleEdge,
    ScheduleEdgeKind,
    ScheduleGraph,
    ScheduleInput,
    ScheduleInputKind,
    ScheduleOutput,
    ScheduleOutputKind,
    ScheduleTask,
    ScheduleValue,
    TaskTiming,
    WorkloadCompilerCoordinator,
    compile_schedule,
)
from tllm.workload import (
    Between,
    State,
    StateScope,
    declare_workload,
    hidden,
    maybe_transition,
    provided,
    read,
    result,
    sampler,
    sink,
    step_id,
    task,
    temporary,
    transition,
    workload,
)


class _Runner:
    device: object = torch.device("cpu")
    model: torch.nn.Module | None = None
    use_cuda_graph: object = False
    compilation_config: CompilationConfigLike | None = None


def _timeline() -> MainWorkloadTimeline:
    capabilities = BackendCapabilities(
        backend_id="vllm-0.27-mrv2",
        runner_family="mrv2",
        sampler_stages=frozenset(),
        native_min_p=False,
        tuple_sampler_return=False,
        split_execute_and_sample=True,
        graph_safety=GraphSafety.EAGER_ONLY,
        hidden_locators=frozenset(
            {HiddenLocatorCapability.BLOCK_OUTPUT_DECODE}
        ),
    )
    return export_main_workload_timeline(
        capabilities,
        graph_mode="eager",
        static_headroom_bytes=1 << 20,
    )


class NativeWorkloadExecutionUnitTest(unittest.TestCase):
    @staticmethod
    def _indexed_hidden_manifest(*paths: tuple[int, str]) -> tuple[ResidualResourceBinding, ...]:
        return tuple(
            ResidualResourceBinding(
                layer_index=layer,
                site=HiddenSite.BLOCK_OUTPUT,
                phase=HiddenPhase.DECODE,
                module_path=path,
            )
            for layer, path in sorted(paths)
        )

    def test_execution_lifecycle_variants_are_total(self) -> None:
        absent = NoExecutionLifecycle()
        self.assertTrue(absent.is_enabled())
        self.assertEqual(absent.read_and_reset_stats(False).loss_count, 0)
        self.assertEqual(absent.read_and_reset_per_request_stats(False), {})
        absent.synchronize()

        synchronized: list[bool] = []
        lifecycle = ExecutionLifecycleBinding(
            stats=lambda sync: TrainingStats(
                loss_avg=1.5 if sync else 0.5,
                loss_count=2,
            ),
            is_enabled=lambda: False,
            synchronize=lambda: synchronized.append(True),
            per_request_stats=PerRequestStats(
                read=lambda sync: {
                    7: TrainingStats(
                        loss_avg=2.5 if sync else 0.25,
                        loss_count=1,
                    )
                }
            ),
        )
        self.assertFalse(lifecycle.is_enabled())
        self.assertEqual(lifecycle.read_and_reset_stats(True).loss_avg, 1.5)
        self.assertEqual(
            lifecycle.read_and_reset_per_request_stats(False)[7].loss_avg,
            0.25,
        )
        lifecycle.synchronize()
        self.assertEqual(synchronized, [True])

    def test_backend_hook_forwards_the_signed_event_without_anchor_lookup(self) -> None:
        events: list[BackendObservableEvent] = []

        class Execution:
            def advance_backend_event(self, event: BackendObservableEvent) -> bool:
                events.append(event)
                return True

        class Runtime:
            workload_execution = Execution()

        common_hooks._advance_native_workload(
            Runtime(),
            BackendObservableEvent.SAMPLE_TOKENS_POST,
        )
        self.assertEqual(events, [BackendObservableEvent.SAMPLE_TOKENS_POST])
        self.assertFalse(hasattr(common_hooks, "_WORKLOAD_ANCHORS_AFTER_EVENT"))

    def test_signed_empty_event_is_a_noop_and_sample_post_alias_executes_once(self) -> None:
        @workload(id="runtime.signed-empty-event")
        class SignedEmptyEvent:
            @task(
                inputs={"value": provided("value")},
                outputs={"output": sink("answer")},
                timing=Between("token.commit", "step.end"),
            )
            def run(self, value: object) -> object:
                return value

        declaration = declare_workload(SignedEmptyEvent())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        empty_advance = next(
            advance
            for advance in compilation.decision.runtime_advance_plan
            if advance.backend_event is BackendObservableEvent.PREPARE_INPUTS_POST
        )
        self.assertEqual(empty_advance.region_ids, ())
        post_advances = tuple(
            advance
            for advance in compilation.decision.runtime_advance_plan
            if advance.backend_event is BackendObservableEvent.SAMPLE_TOKENS_POST
        )
        self.assertEqual(len(post_advances), 2)
        self.assertEqual(post_advances[0].anchor_id, "token.commit")
        self.assertEqual(len(post_advances[0].region_ids), 1)
        self.assertEqual(post_advances[1].anchor_id, "step.end")
        self.assertEqual(post_advances[1].region_ids, ())

        state = ResidualRuntimeState()
        bindings = WorkloadExecutionBindings(
            declarations=(declaration,),
            ports=RuntimePortRegistry(
                inputs={
                    "value": RuntimeInputBinding(
                        lambda _step_id: torch.tensor(1)
                    )
                },
                outputs={
                    "answer": RuntimeOutputBinding(
                        lambda _value, _step_id: None
                    )
                },
            ),
        )
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=bindings,
            )
            execution = state.workload_execution
            self.assertIsNotNone(execution)
            assert execution is not None
            execution.prepare_step(1)
            execute_nodes_impl = type(execution.executor).execute_nodes
            with mock.patch.object(
                type(execution.executor),
                "execute_nodes",
                autospec=True,
                side_effect=execute_nodes_impl,
            ) as execute_nodes:
                self.assertFalse(
                    execution.advance_backend_event(
                        BackendObservableEvent.PREPARE_INPUTS_POST
                    )
                )
                execute_nodes.assert_not_called()
                # token.commit and step.end share SAMPLE_TOKENS_POST.  Its
                # Region must run only on the first typed event dispatch.
                self.assertTrue(
                    execution.advance_backend_event(
                        BackendObservableEvent.SAMPLE_TOKENS_POST
                    )
                )
                self.assertFalse(
                    execution.advance_backend_event(
                        BackendObservableEvent.SAMPLE_TOKENS_POST
                    )
                )
                self.assertEqual(execute_nodes.call_count, 1)
            execution.close()

    def test_path_hidden_binding_requires_a_signed_backend_alias(self) -> None:
        binding = RegionInputBinding(
            binding_id="hidden",
            kind=RegionInputKind.HIDDEN,
            source="hidden:model.layers.0:block_output:decode",
            consumers=(RegionPort(task_id="run", port_name="hidden"),),
            layer="model.layers.0",
            layer_kind=HiddenLayerKind.PATH,
            site="block_output",
            phase="decode",
        )
        runtime = ResidualRuntimeState()
        runtime.decode_count = 1
        # A same-named capture buffer is not a backend binding. The signed
        # workload resource must have resolved this path before execution.
        runtime.tap_decode_hidden = {
            "model.layers.0": torch.ones((1, 4), dtype=torch.float32)
        }

        with self.assertRaisesRegex(RuntimeError, "explicit backend alias"):
            _hidden_value(runtime, binding)

    def test_path_hidden_binding_owns_an_immutable_backend_snapshot(self) -> None:
        binding = RegionInputBinding(
            binding_id="hidden",
            kind=RegionInputKind.HIDDEN,
            source="hidden:source:block_output:decode",
            consumers=(RegionPort(task_id="run", port_name="hidden"),),
            layer="source",
            layer_kind=HiddenLayerKind.PATH,
            site="block_output",
            phase="decode",
        )
        hidden = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        runtime = ResidualRuntimeState()
        runtime.decode_count = 1
        runtime.native_hidden_path_aliases = {"source": "model.layers.0"}
        runtime.tap_decode_hidden = {"model.layers.0": hidden}

        staged = _hidden_value(runtime, binding)

        self.assertIsInstance(staged, torch.Tensor)
        assert isinstance(staged, torch.Tensor)
        self.assertTrue(torch.equal(staged, hidden))
        self.assertNotEqual(
            staged.untyped_storage().data_ptr(), hidden.untyped_storage().data_ptr()
        )
        hidden.add_(1)
        torch.testing.assert_close(
            staged,
            torch.arange(4, dtype=torch.float32).reshape(1, 4),
        )

    def test_hidden_binding_stages_inference_tensor_for_region_autograd(self) -> None:
        locator = ResidualLocator(0, "block_output", "decode")
        binding = RegionInputBinding(
            binding_id="hidden",
            kind=RegionInputKind.HIDDEN,
            source="hidden:0:block_output:decode",
            consumers=(RegionPort(task_id="run", port_name="hidden"),),
            layer="0",
            layer_index=0,
            layer_kind=HiddenLayerKind.INDEX,
            site="block_output",
            phase="decode",
        )
        with torch.inference_mode():
            hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        runtime = ResidualRuntimeState()
        runtime.decode_count = 1
        runtime.residual_bindings = {
            "layers.0": ResidualPathBinding(locator, "layers.0")
        }
        runtime.tap_decode_hidden = {"layers.0": hidden}

        staged = _hidden_value(runtime, binding)

        self.assertIsInstance(staged, torch.Tensor)
        assert isinstance(staged, torch.Tensor)
        self.assertIs(binding.site, HiddenSite.BLOCK_OUTPUT)
        self.assertIs(binding.phase, HiddenPhase.DECODE)
        self.assertFalse(torch.is_inference(staged))
        self.assertTrue(torch.equal(staged, hidden[:1]))

    def test_captured_region_honors_a_disabled_native_workload(self) -> None:
        class DisabledStore:
            def __init__(self) -> None:
                self.committed: dict[str, object] | None = None

            def is_enabled(self) -> bool:
                return False

            def commit_outputs(self, _artifact, _context, outputs) -> None:
                self.committed = dict(outputs)

        artifact = RegionCallableArtifact(
            region_id="region-test",
            artifact_key="0" * 64,
            input_keys=(),
            output_keys=("output",),
            callable=lambda: (torch.tensor(1.0),),
            compiled=False,
        )
        store = DisabledStore()
        graph = _RegionCudaGraph(artifact, cast(Any, store))
        context = PhysicalExecutionContext(
            step_id=1,
            plan_epoch=1,
            node=PhysicalNodePlan(
                node_key=NodeKey("test", "captured"),
                executor=ExecutorKey("test", "captured"),
                stream=StreamClass.DEFAULT,
            ),
            kind=PhysicalExecutionKind.REPLAY,
            stream=object(),
            tensors=(),
            leases=(),
        )

        graph.replay_context(context)

        self.assertEqual(store.committed, {"output": SKIPPED})

    def test_region_cuda_graph_shape_miss_queues_signed_specialization_request(self) -> None:
        @workload(id="runtime.capture-shape-miss")
        class Captured:
            @task(
                inputs={
                    "left": provided("left"),
                    "right": provided("right"),
                },
                outputs={"output": sink("answer")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, left: object, right: object) -> object:
                return cast(torch.Tensor, left) + cast(torch.Tensor, right)

        declaration = declare_workload(Captured())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        region = compilation.decision.execution_regions[0]
        input_keys = tuple(
            reversed(tuple(binding.binding_id for binding in region.input_bindings))
        )
        artifact = RegionCallableArtifact(
            region_id=region.region_id,
            artifact_key=region.artifact_key,
            input_keys=input_keys,
            output_keys=tuple(
                binding.binding_id for binding in region.output_bindings
            ),
            callable=lambda left, right: (left + right,),
            compiled=False,
            input_bindings=region.input_bindings,
            output_bindings=region.output_bindings,
            effects=region.effects,
        )
        assert compilation.physical is not None
        context = PhysicalExecutionContext(
            step_id=3,
            plan_epoch=compilation.physical.plan_epoch,
            node=compilation.physical.nodes[0],
            kind=PhysicalExecutionKind.REPLAY,
            stream=object(),
            tensors=(),
            leases=(),
        )

        class Store:
            def __init__(self) -> None:
                self.commits: list[dict[str, object]] = []

            def is_enabled(self) -> bool:
                return True

            def resolve_inputs(self, _artifact, _context) -> dict[str, object]:
                return {key: object() for key in artifact.input_keys}

            def commit_outputs(self, _artifact, _context, outputs) -> None:
                self.commits.append(dict(outputs))

        store = Store()
        queue = _NativeShapeSpecializationQueue()

        def request_specialization(
            request_context: PhysicalExecutionContext,
            inputs: Mapping[str, torch.Tensor],
        ) -> "RegionShapeSpecializationRequest":
            request = _region_shape_specialization_request(
                compilation,
                artifact,
                request_context,
                inputs,
            )
            queue.enqueue(request)
            return request

        graph = _RegionCudaGraph(
            artifact,
            cast(Any, store),
            request_specialization,
        )
        initial_inputs = {
            input_keys[0]: torch.empty((1, 4), dtype=torch.float32),
            input_keys[1]: torch.empty((1, 4), dtype=torch.float16),
        }

        def fake_capture(
            _graph: _RegionCudaGraph,
            inputs: Mapping[str, torch.Tensor],
        ) -> None:
            graph._graph = cast(Any, object())
            graph._stream = cast(Any, object())
            graph._input_specs = {
                key: (tuple(value.shape), value.dtype, value.device)
                for key, value in inputs.items()
            }
            graph._outputs = (inputs[input_keys[0]],)

        # The first capture fixes its observed Tensor contract. It is not a
        # specialization miss and must leave the cold-path queue empty.
        with mock.patch.object(
            _RegionCudaGraph,
            "_tensor_inputs",
            return_value=initial_inputs,
        ), mock.patch.object(
            _RegionCudaGraph,
            "_capture",
            autospec=True,
            side_effect=fake_capture,
        ), mock.patch(
            "tllm.contracts.shape_specialization._normalize_json",
            side_effect=lambda payload: payload,
        ):
            graph.replay_context(context)
            self.assertEqual(queue.peek(), ())
            self.assertEqual(len(store.commits), 1)

            store.commits.clear()
            mismatch_inputs = {
                input_keys[0]: torch.empty((2, 4), dtype=torch.float32),
                input_keys[1]: torch.empty((1, 4), dtype=torch.float16),
            }
            with mock.patch.object(
                _RegionCudaGraph,
                "_tensor_inputs",
                return_value=mismatch_inputs,
            ):
                with self.assertRaises(ShapeSpecializationMiss) as miss:
                    graph.preflight_context(context)
                with self.assertRaises(ShapeSpecializationMiss) as repeated:
                    graph.preflight_context(context)
                with self.assertRaises(ShapeSpecializationMiss) as replayed:
                    graph.replay_context(context)

        request = miss.exception.request
        self.assertEqual(request.base_cache_key, compilation.cache_key)
        self.assertEqual(request.base_signature, compilation.decision.signature)
        self.assertEqual(request.observed_plan_epoch, compilation.physical.plan_epoch)
        self.assertEqual(request.region_id, region.region_id)
        self.assertEqual(request.artifact_key, region.artifact_key)
        self.assertEqual(
            tuple(item.binding_id for item in request.inputs),
            tuple(sorted(input_keys)),
        )
        self.assertEqual(
            {item.binding_id: item.shape for item in request.inputs},
            {input_keys[0]: (2, 4), input_keys[1]: (1, 4)},
        )
        self.assertIs(repeated.exception.request, request)
        self.assertIs(replayed.exception.request, request)
        self.assertEqual(queue.peek(), (request,))
        self.assertEqual(store.commits, [])

        execution = NativeWorkloadExecution(
            compilation=compilation,
            executor=cast(Any, object()),
            completion_anchor=compilation.decision.completion_anchor,
            _owned_consumers=(),
            _shape_specializations=queue,
        )
        self.assertEqual(execution.peek_shape_specialization_requests(), (request,))
        self.assertEqual(execution.drain_shape_specialization_requests(), (request,))
        self.assertEqual(execution.peek_shape_specialization_requests(), ())

    def test_pure_two_task_region_executes_as_one_authoritative_artifact(self) -> None:
        trace: list[str] = []
        outputs: list[tuple[int, torch.Tensor]] = []

        @workload(id="runtime.fused")
        class Fused:
            @task(
                inputs={"value": provided("value")},
                outputs={"middle": temporary()},
                timing=Between("model.execute", "step.end"),
            )
            def produce(self, value: object) -> object:
                trace.append("produce")
                assert isinstance(value, torch.Tensor)
                return value + 1

            @task(
                inputs={"middle": result("produce", "middle")},
                outputs={"output": sink("answer")},
                timing=Between("model.execute", "step.end"),
            )
            def consume(self, middle: object) -> object:
                trace.append("consume")
                assert isinstance(middle, torch.Tensor)
                return middle * 2

        declaration = declare_workload(Fused())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        state = ResidualRuntimeState()
        bindings = WorkloadExecutionBindings(
            declarations=(declaration,),
            ports=RuntimePortRegistry(
                inputs={
                    "value": RuntimeInputBinding(
                        lambda step_id: torch.tensor([float(step_id)])
                    )
                },
                outputs={
                    "answer": RuntimeOutputBinding(
                        lambda value, step_id: outputs.append(
                            (step_id, cast(torch.Tensor, value))
                        )
                    )
                },
            ),
        )

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=bindings,
            )
            execution = state.workload_execution
            self.assertIsNotNone(execution)
            assert execution is not None
            self.assertEqual(len(compilation.decision.execution_regions), 1)
            region = compilation.decision.execution_regions[0]
            self.assertEqual(
                tuple(call.task_id for call in region.task_invocations),
                ("produce", "consume"),
            )
            self.assertEqual(
                region.task_invocations[0].outputs[0].value_binding_id,
                region.task_invocations[1].inputs[0].value_binding_id,
            )

            residual_runtime.prepare_physical_workgraph_step(3)
            anchor = region.placement_anchor
            self.assertTrue(residual_runtime.advance_workload_anchor(anchor))
            self.assertFalse(residual_runtime.advance_workload_anchor(anchor))

        self.assertEqual(trace, ["produce", "consume"])
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0][0], 3)
        torch.testing.assert_close(outputs[0][1], torch.tensor([8.0]))
        self.assertEqual(execution.executor.stats.execute_count, 1)
        self.assertEqual(execution.executor.stats.eager_count, 1)

    def test_regions_execute_at_distinct_anchors_and_retain_cross_region_value(self) -> None:
        trace: list[str] = []
        outputs: list[torch.Tensor] = []

        @workload(id="runtime.staged")
        class Staged:
            @task(
                inputs={"value": provided("value")},
                outputs={"middle": temporary()},
                timing=Between("model.execute", "logits.post"),
            )
            def produce(self, value: object) -> object:
                trace.append("produce")
                assert isinstance(value, torch.Tensor)
                return value + 1

            @task(
                inputs={"middle": result("produce", "middle")},
                outputs={"output": sink("answer")},
                timing=Between("token.commit", "step.end"),
            )
            def consume(self, middle: object) -> object:
                trace.append("consume")
                assert isinstance(middle, torch.Tensor)
                return middle * 2

        declaration = declare_workload(Staged())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        state = ResidualRuntimeState()
        bindings = WorkloadExecutionBindings(
            declarations=(declaration,),
            ports=RuntimePortRegistry(
                inputs={
                    "value": RuntimeInputBinding(
                        lambda _step_id: torch.tensor([4.0])
                    )
                },
                outputs={
                    "answer": RuntimeOutputBinding(
                        lambda value, _step_id: outputs.append(
                            cast(torch.Tensor, value)
                        )
                    )
                },
            ),
        )

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=bindings,
            )
            execution = state.workload_execution
            self.assertIsNotNone(execution)
            assert execution is not None
            self.assertEqual(len(compilation.decision.execution_regions), 2)
            self.assertEqual(len(compilation.decision.materializations), 1)
            first, second = compilation.decision.execution_regions
            materialization = compilation.decision.materializations[0]
            self.assertEqual(
                tuple(binding.kind for binding in first.input_bindings),
                ("provided",),
            )
            self.assertEqual(
                tuple(binding.kind for binding in first.output_bindings),
                ("materialized",),
            )
            self.assertEqual(
                first.output_bindings[0].binding_id,
                materialization.materialization_id,
            )
            self.assertEqual(
                second.input_bindings[0].binding_id,
                materialization.materialization_id,
            )
            self.assertEqual(second.output_bindings[0].kind, "sink")

            residual_runtime.prepare_physical_workgraph_step(5)
            self.assertTrue(residual_runtime.advance_workload_anchor(first.placement_anchor))
            self.assertEqual(trace, ["produce"])
            self.assertEqual(outputs, [])
            self.assertEqual(execution.retained_value_count, 1)

            self.assertTrue(residual_runtime.advance_workload_anchor(second.placement_anchor))
            self.assertEqual(trace, ["produce", "consume"])
            self.assertEqual(execution.retained_value_count, 0)

        self.assertEqual(len(outputs), 1)
        torch.testing.assert_close(outputs[0], torch.tensor([10.0]))
        self.assertEqual(execution.executor.stats.execute_count, 2)
        self.assertEqual(execution.executor.stats.fence_record_count, 1)
        self.assertEqual(execution.executor.stats.fence_wait_count, 1)

    def test_materialization_table_is_exact_signed_region_boundary(self) -> None:
        @workload(id="runtime.materialization-table")
        class Staged:
            @task(
                inputs={"value": provided("value")},
                outputs={"middle": temporary()},
                timing=Between("model.execute", "logits.post"),
            )
            def produce(self, value: object) -> object:
                return value

            @task(
                inputs={"middle": result("produce", "middle")},
                outputs={"output": sink("answer")},
                timing=Between("token.commit", "step.end"),
            )
            def consume(self, middle: object) -> object:
                return middle

        declaration = declare_workload(Staged())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(memory_budget_bytes=1 << 20, compile_policy="eager"),
        ).compile(_timeline())
        artifacts = materialize_region_artifacts(
            (declaration,), compilation.decision, compilation.options
        )
        source, target = compilation.decision.execution_regions
        materialization = compilation.decision.materializations[0]

        table = lower_signed_materialization_bindings(
            compilation.graph, compilation.decision, artifacts
        )
        binding = table.input_binding_for(
            target.region_id, materialization.materialization_id
        )
        self.assertEqual(binding.producer_region, source.region_id)
        self.assertEqual(binding.consumer_region, target.region_id)
        self.assertEqual(binding.value_id, materialization.value_id)
        self.assertEqual(binding.version, materialization.version)
        self.assertEqual(binding.live_start, source.placement_anchor)
        self.assertEqual(binding.live_end, target.placement_anchor)

        missing_input = (
            artifacts[0],
            replace(artifacts[1], input_bindings=()),
        )
        with self.assertRaisesRegex(RuntimeError, "materialized input bindings"):
            lower_signed_materialization_bindings(
                compilation.graph, compilation.decision, missing_input
            )

        extra_input = (
            artifacts[0],
            replace(
                artifacts[1],
                input_bindings=artifacts[1].input_bindings
                + (replace(artifacts[1].input_bindings[0], binding_id="unsigned-value"),),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "materialized input bindings"):
            lower_signed_materialization_bindings(
                compilation.graph, compilation.decision, extra_input
            )

        missing_output = (
            replace(artifacts[0], output_bindings=()),
            artifacts[1],
        )
        with self.assertRaisesRegex(RuntimeError, "materialized output bindings"):
            lower_signed_materialization_bindings(
                compilation.graph, compilation.decision, missing_output
            )

        wrong_version = replace(
            compilation.decision,
            materializations=(replace(materialization, version=materialization.version + 1),),
        )
        with self.assertRaisesRegex(RuntimeError, "version"):
            lower_signed_materialization_bindings(
                compilation.graph, wrong_version, artifacts
            )

        wrong_producer = replace(
            compilation.decision,
            execution_regions=(
                replace(
                    source,
                    output_bindings=(
                        replace(
                            source.output_bindings[0],
                            producer=RegionPort("consume", "middle"),
                        ),
                    ),
                ),
                target,
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "wrong producer"):
            lower_signed_materialization_bindings(
                compilation.graph,
                wrong_producer,
                (
                    replace(artifacts[0], output_bindings=wrong_producer.execution_regions[0].output_bindings),
                    artifacts[1],
                ),
            )

        wrong_lifetime = replace(
            compilation.decision,
            materializations=(replace(materialization, live_end="model.execute"),),
        )
        with self.assertRaisesRegex(RuntimeError, "lifetime"):
            lower_signed_materialization_bindings(
                compilation.graph, wrong_lifetime, artifacts
            )

    def test_fanout_materializations_release_each_signed_boundary_independently(self) -> None:
        trace: list[str] = []

        @workload(id="runtime.materialization-fanout")
        class Fanout:
            @task(
                inputs={"value": provided("value")},
                outputs={"middle": temporary()},
                timing=Between("model.execute", "logits.post"),
            )
            def produce(self, value: object) -> object:
                trace.append("produce")
                return value

            @task(
                inputs={"middle": result("produce", "middle")},
                outputs={"output": sink("first")},
                timing=Between("sampler.pre_draw", "token.commit"),
            )
            def first(self, middle: object) -> object:
                trace.append("first")
                return middle

            @task(
                inputs={"middle": result("produce", "middle")},
                outputs={"output": sink("second")},
                timing=Between("token.commit", "step.end"),
            )
            def second(self, middle: object) -> object:
                trace.append("second")
                return middle

        declaration = declare_workload(Fanout())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(memory_budget_bytes=1 << 20, compile_policy="eager"),
        ).compile(_timeline())
        self.assertEqual(len(compilation.decision.materializations), 2)
        self.assertEqual(
            len({item.materialization_id for item in compilation.decision.materializations}),
            2,
        )
        table = lower_signed_materialization_bindings(
            compilation.graph,
            compilation.decision,
            materialize_region_artifacts(
                (declaration,), compilation.decision, compilation.options
            ),
        )
        self.assertEqual(
            {
                table.input_binding_for(
                    materialization.consumer_region,
                    materialization.materialization_id,
                ).consumer_region
                for materialization in compilation.decision.materializations
            },
            {
                materialization.consumer_region
                for materialization in compilation.decision.materializations
            },
        )

        state = ResidualRuntimeState()
        bindings = WorkloadExecutionBindings(
            declarations=(declaration,),
            ports=RuntimePortRegistry(
                inputs={
                    "value": RuntimeInputBinding(
                        lambda _step_id: torch.tensor([1.0])
                    )
                },
                outputs={
                    "first": RuntimeOutputBinding(
                        lambda _value, _step_id: None
                    ),
                    "second": RuntimeOutputBinding(
                        lambda _value, _step_id: None
                    ),
                },
            ),
        )
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(compilation, bindings=bindings)
            execution = state.workload_execution
            assert execution is not None
            residual_runtime.prepare_physical_workgraph_step(7)
            for position, region in enumerate(compilation.decision.execution_regions):
                self.assertTrue(
                    residual_runtime.advance_workload_anchor(region.placement_anchor)
                )
                self.assertEqual(execution.retained_value_count, 2 - min(position, 2))

        self.assertEqual(trace, ["produce", "first", "second"])

    def test_versioned_materializations_execute_without_cross_version_aliasing(self) -> None:
        graph = ScheduleGraph(
            workload_key="runtime.versioned-materializations",
            tasks=(
                ScheduleTask(
                    "produce",
                    timing=TaskTiming(after="model.execute", before="logits.post"),
                    outputs=(
                        ScheduleOutput("v1", ScheduleOutputKind.TEMPORARY, "v1"),
                        ScheduleOutput("v2", ScheduleOutputKind.TEMPORARY, "v2"),
                    ),
                ),
                ScheduleTask(
                    "consume_v1",
                    timing=TaskTiming(after="token.commit", before="step.end"),
                    inputs=(
                        ScheduleInput("value", ScheduleInputKind.RESULT, "produce", "v1"),
                    ),
                ),
                ScheduleTask(
                    "consume_v2",
                    timing=TaskTiming(after="token.commit", before="step.end"),
                    inputs=(
                        ScheduleInput("value", ScheduleInputKind.RESULT, "produce", "v2"),
                    ),
                ),
            ),
            edges=(
                ScheduleEdge("produce", "consume_v1", ScheduleEdgeKind.DATA),
                ScheduleEdge("produce", "consume_v2", ScheduleEdgeKind.DATA),
            ),
            values=(
                ScheduleValue(
                    "hidden",
                    version=1,
                    producer="produce",
                    producer_output="v1",
                    consumers=("consume_v1",),
                ),
                ScheduleValue(
                    "hidden",
                    version=2,
                    producer="produce",
                    producer_output="v2",
                    consumers=("consume_v2",),
                ),
            ),
        )
        decision = compile_schedule(
            graph, _timeline(), CompileOptions(memory_budget_bytes=1 << 20)
        )
        producer_region = next(
            region for region in decision.execution_regions if region.tasks == ("produce",)
        )
        consumer_region = next(
            region
            for region in decision.execution_regions
            if {"consume_v1", "consume_v2"}.issubset(region.tasks)
        )
        producer = RegionCallableArtifact(
            region_id=producer_region.region_id,
            artifact_key=producer_region.artifact_key,
            input_keys=(),
            output_keys=tuple(
                binding.binding_id for binding in producer_region.output_bindings
            ),
            callable=lambda: ("v1", "v2"),
            compiled=False,
            input_bindings=producer_region.input_bindings,
            output_bindings=producer_region.output_bindings,
            effects=(),
        )
        consumer = RegionCallableArtifact(
            region_id=consumer_region.region_id,
            artifact_key=consumer_region.artifact_key,
            input_keys=tuple(
                binding.binding_id for binding in consumer_region.input_bindings
            ),
            output_keys=(),
            callable=lambda *_values: (),
            compiled=False,
            input_bindings=consumer_region.input_bindings,
            output_bindings=consumer_region.output_bindings,
            effects=(),
        )
        artifacts = (producer, consumer)
        materializations = lower_signed_materialization_bindings(
            graph, decision, artifacts
        )
        store = _RegionValueStore(
            runtime=ResidualRuntimeState(),
            providers={},
            sinks={},
            artifacts=artifacts,
            materializations=materializations,
            state_versions=StateVersionRegistry((), {}, ()),
            lifecycle=NoExecutionLifecycle(),
        )
        context = PhysicalExecutionContext(
            step_id=9,
            plan_epoch=1,
            node=PhysicalNodePlan(
                node_key=NodeKey("test", "versioned-materializations"),
                executor=ExecutorKey("test", "versioned-materializations"),
                stream=StreamClass.DEFAULT,
            ),
            kind=PhysicalExecutionKind.EAGER,
            stream=object(),
            tensors=(),
            leases=(),
        )
        store.prepare_step(9)
        store.execute(producer, context)
        self.assertEqual(store.retained_value_count, 2)
        self.assertEqual(set(store._frames[9].values.values()), {"v1", "v2"})
        store.execute(consumer, context)
        self.assertEqual(store.retained_value_count, 0)

    def test_physical_region_binding_is_invariant_to_legal_node_reordering(self) -> None:
        trace: list[str] = []

        @workload(id="runtime.unordered")
        class Unordered:
            @task(
                inputs={"value": provided("left")},
                outputs={"output": sink("left")},
                timing=Between("model.execute", "logits.post"),
            )
            def left(self, value: object) -> object:
                trace.append("left")
                return value

            @task(
                inputs={"value": provided("right")},
                outputs={"output": sink("right")},
                timing=Between("token.commit", "step.end"),
            )
            def right(self, value: object) -> object:
                trace.append("right")
                return value

        declaration = declare_workload(Unordered())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        physical = compilation.physical
        self.assertIsNotNone(physical)
        assert physical is not None
        self.assertEqual(len(physical.nodes), 2)
        self.assertFalse(any(node.fences for node in physical.nodes))
        reordered = replace(
            physical,
            nodes=tuple(reversed(physical.nodes)),
            anchor_placements=tuple(reversed(physical.anchor_placements)),
        )
        reordered_compilation = replace(compilation, physical=reordered)
        regions = compilation.decision.execution_regions
        self.assertEqual(len(regions), 2)
        bindings = WorkloadExecutionBindings(
            declarations=(declaration,),
            ports=RuntimePortRegistry(
                inputs={
                    "left": RuntimeInputBinding(
                        lambda _step_id: torch.tensor(1)
                    ),
                    "right": RuntimeInputBinding(
                        lambda _step_id: torch.tensor(2)
                    ),
                },
                outputs={
                    "left": RuntimeOutputBinding(
                        lambda _value, _step_id: None
                    ),
                    "right": RuntimeOutputBinding(
                        lambda _value, _step_id: None
                    ),
                },
            ),
        )
        state = ResidualRuntimeState()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                reordered_compilation,
                bindings=bindings,
            )
            execution = state.workload_execution
            self.assertIsNotNone(execution)
            assert execution is not None
            self.assertEqual(
                execution.completion_anchor,
                compilation.decision.completion_anchor,
            )

            residual_runtime.prepare_physical_workgraph_step(1)
            self.assertTrue(
                execution.advance_backend_event(
                    BackendObservableEvent.EXECUTE_MODEL_PRE
                )
            )
            self.assertEqual(trace, ["left"])
            self.assertTrue(
                execution.advance_backend_event(
                    BackendObservableEvent.SAMPLE_TOKENS_POST
                )
            )
            self.assertFalse(
                execution.advance_backend_event(
                    BackendObservableEvent.SAMPLE_TOKENS_POST
                )
            )
            execution._event_advances = {}
            with self.assertRaisesRegex(RuntimeError, "not signed"):
                execution.advance_backend_event(BackendObservableEvent.SAMPLER_PRE_DRAW)
            self.assertFalse(execution.advance_anchor("unsigned.anchor"))

        self.assertEqual(trace, ["left", "right"])

    def test_zero_node_sampler_consumer_still_joins_pending_side_work(self) -> None:
        """A sampler consumer must fence even when it advances zero Regions.

        The sampler provider reached via dispatch_sampler_pre_draw consumes
        side-Region outputs at an anchor that typically advances no Region;
        skipping the join there re-opens the pre-draw consumption race.
        """

        @workload(id="runtime.zero-node-join")
        class ZeroNode:
            @task(
                outputs={"value": temporary()},
                timing=Between("model.execute", "logits.post"),
            )
            def produce(self) -> torch.Tensor:
                return torch.ones(4)

            @task(
                inputs={"value": result("produce", "value")},
                outputs={
                    "guidance": sampler.hidden_reweight(beta=1.0),
                },
                timing=Between("logits.post", "sampler.pre_draw"),
            )
            def publish(self, value: torch.Tensor) -> torch.Tensor:
                return value

        declaration = declare_workload(ZeroNode())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        bindings = WorkloadExecutionBindings(
            declarations=(declaration,),
            ports=RuntimePortRegistry(
                outputs={
                    "guidance": RuntimeOutputBinding(
                        lambda _value, _step_id: None
                    )
                }
            ),
        )
        state = ResidualRuntimeState()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(compilation, bindings=bindings)
            execution = state.workload_execution
            self.assertIsNotNone(execution)
            assert execution is not None
            residual_runtime.prepare_physical_workgraph_step(1)

            class SideStream:
                pass

            waited: list[SideStream] = []

            class MainStream:
                def wait_stream(self, stream: SideStream) -> None:
                    waited.append(stream)

            side_stream = SideStream()
            execution._side_streams = (side_stream,)
            execution._main_stream = MainStream()
            execution._side_work_pending = True
            self.assertIn(
                BackendObservableEvent.SAMPLER_PRE_DRAW,
                execution._event_wait_after,
            )
            self.assertFalse(
                execution.advance_backend_event(
                    BackendObservableEvent.SAMPLER_PRE_DRAW
                )
            )
            self.assertEqual(waited, [side_stream])
            self.assertFalse(execution._side_work_pending)
            execution.finish_step(1)

    def test_side_work_crosses_retirement_and_empty_events_before_next_state_reader(
        self,
    ) -> None:
        trace: list[str] = []
        versions: dict[Hashable, int] = {None: 0}

        @workload(id="runtime.side-lifecycle")
        class SideLifecycle:
            state = State(
                scope=StateScope.REPLICA,
                parameters="model",
                transition="manual",
            )

            def __init__(self) -> None:
                self.weight = torch.nn.Parameter(torch.tensor(2.0))

            @task(
                inputs={"current_step": step_id()},
                outputs={"prediction": temporary()},
                effects=(read(state),),
                timing=Between("model.execute", "sampler.pre_draw"),
            )
            def predict(self, current_step: int) -> torch.Tensor:
                trace.append(f"read:{current_step}")
                return self.weight * (current_step + 1)

            @task(
                inputs={"prediction": result("predict", "prediction")},
                effects=(transition(state),),
                timing=Between("token.commit", "next.prepare"),
            )
            def update(self, prediction: torch.Tensor) -> None:
                trace.append("update")
                prediction.square().backward()
                versions[None] += 1

        declaration = declare_workload(SideLifecycle())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
                allow_side_stream=True,
            ),
        ).compile(_timeline())
        streams = {
            task_name: region.stream.value
            for region in compilation.decision.execution_regions
            for task_name in region.tasks
        }
        self.assertEqual(
            streams,
            {"predict": "default", "update": "side_forward"},
        )
        schedule_state = compilation.graph.states[0]
        bindings = WorkloadExecutionBindings(
            declarations=(declaration,),
            state_versions={
                schedule_state.resource: StateVersionBinding(
                    schedule_state,
                    lambda _step_id: dict(versions),
                )
            },
        )
        state = ResidualRuntimeState()

        def join_pending_side_work(execution: NativeWorkloadExecution) -> None:
            if execution._side_work_pending:
                trace.append("join")
                execution._side_work_pending = False

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(
                NativeWorkloadExecution,
                "_join_side_streams_locked",
                autospec=True,
                side_effect=join_pending_side_work,
            ),
        ):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=bindings,
            )
            execution = state.workload_execution
            self.assertIsNotNone(execution)
            assert execution is not None
            self.assertEqual(
                execution._event_nodes[
                    BackendObservableEvent.PREPARE_INPUTS_POST
                ],
                (),
            )
            self.assertIn(
                BackendObservableEvent.EXECUTE_MODEL_POST,
                execution._event_wait_before,
            )
            self.assertNotIn(
                BackendObservableEvent.PREPARE_INPUTS_POST,
                execution._event_wait_before,
            )

            execution.prepare_step(1)
            self.assertTrue(
                execution.advance_backend_event(
                    BackendObservableEvent.EXECUTE_MODEL_POST
                )
            )
            self.assertTrue(
                execution.advance_backend_event(
                    BackendObservableEvent.SAMPLE_TOKENS_POST
                )
            )
            self.assertTrue(execution._side_work_pending)

            execution.finish_step(1)
            self.assertTrue(execution._side_work_pending)
            self.assertEqual(trace, ["read:1", "update"])

            execution.prepare_step(2)
            self.assertFalse(
                execution.advance_backend_event(
                    BackendObservableEvent.PREPARE_INPUTS_POST
                )
            )
            self.assertTrue(execution._side_work_pending)
            self.assertEqual(trace, ["read:1", "update"])

            self.assertTrue(
                execution.advance_backend_event(
                    BackendObservableEvent.EXECUTE_MODEL_POST
                )
            )
            self.assertFalse(execution._side_work_pending)
            self.assertEqual(
                trace,
                ["read:1", "update", "join", "read:2"],
            )

            self.assertTrue(
                execution.advance_backend_event(
                    BackendObservableEvent.SAMPLE_TOKENS_POST
                )
            )
            self.assertTrue(execution._side_work_pending)
            execution.synchronize()
            self.assertFalse(execution._side_work_pending)

            execution.finish_step(2)
            execution.prepare_step(3)
            self.assertTrue(
                execution.advance_backend_event(
                    BackendObservableEvent.EXECUTE_MODEL_POST
                )
            )
            self.assertTrue(
                execution.advance_backend_event(
                    BackendObservableEvent.SAMPLE_TOKENS_POST
                )
            )
            self.assertTrue(execution._side_work_pending)
            execution.close()
            self.assertFalse(execution._side_work_pending)

        self.assertEqual(trace.count("join"), 3)
        self.assertEqual(versions, {None: 3})

    def test_side_autograd_materialization_records_stream_ownership_before_launch(
        self,
    ) -> None:
        trace: list[str] = []
        recorded: list[torch.Tensor] = []
        producer_streams: list[torch.cuda.Stream] = []

        @workload(id="runtime.side-autograd-ownership")
        class SideAutogradOwnership:
            def __init__(self) -> None:
                self.weight = torch.nn.Parameter(torch.tensor(2.0))

            @task(
                inputs={"current_step": step_id()},
                outputs={"prediction": temporary()},
                timing=Between("model.execute", "sampler.pre_draw"),
            )
            def predict(self, current_step: int) -> torch.Tensor:
                trace.append("predict")
                prediction = self.weight * (current_step + 1)
                return prediction

            @task(
                inputs={"prediction": result("predict", "prediction")},
                timing=Between("token.commit", "next.prepare"),
            )
            def train(self, prediction: torch.Tensor) -> None:
                trace.append("train")
                prediction.square().backward()

        declaration = declare_workload(SideAutogradOwnership())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
                allow_side_stream=True,
            ),
        ).compile(_timeline())
        self.assertEqual(
            {
                task_name: region.stream.value
                for region in compilation.decision.execution_regions
                for task_name in region.tasks
            },
            {"predict": "default", "train": "side_forward"},
        )

        state = ResidualRuntimeState()
        def record_owners(
            values: tuple[torch.Tensor, ...],
            _stream: torch.cuda.Stream,
        ) -> None:
            if not producer_streams:
                producer_streams.append(_stream)
            elif values and _stream is not producer_streams[0]:
                trace.append("record")
                recorded.extend(values)

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch(
                "tllm.runtime.workload_execution.torch_record_stream_owners",
                side_effect=record_owners,
            ),
        ):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=WorkloadExecutionBindings(
                    declarations=(declaration,)
                ),
            )
            execution = state.workload_execution
            self.assertIsNotNone(execution)
            assert execution is not None
            execution.prepare_step(1)
            self.assertTrue(
                execution.advance_backend_event(
                    BackendObservableEvent.EXECUTE_MODEL_POST
                )
            )
            self.assertTrue(
                execution.advance_backend_event(
                    BackendObservableEvent.SAMPLE_TOKENS_POST
                )
            )
            self.assertTrue(execution._side_work_pending)
            execution.finish_step(1)
            execution.synchronize()
        self.assertEqual(trace, ["predict", "record", "train"])
        self.assertEqual(len(recorded), 1)
        self.assertTrue(recorded[0].requires_grad)

    def test_default_state_mutation_waits_for_prior_side_read(self) -> None:
        @workload(id="runtime.side-read-default-write")
        class SideReadDefaultWrite:
            state = State(
                scope=StateScope.REPLICA,
                parameters="model",
                transition="manual",
            )

            @task(
                outputs={"value": temporary()},
                timing=Between("model.execute", "sampler.pre_draw"),
            )
            def produce(self) -> torch.Tensor:
                return torch.ones(1)

            @task(
                inputs={"value": result("produce", "value")},
                effects=(read(state),),
                timing=Between("sampler.pre_draw", "token.commit"),
            )
            def observe(self, value: torch.Tensor) -> None:
                _ = value.square()

            @task(
                inputs={"trigger": provided("trigger")},
                effects=(transition(state),),
                timing=Between("token.commit", "next.prepare"),
            )
            def mutate(self, trigger: torch.Tensor) -> None:
                _ = trigger

        compilation = WorkloadCompilerCoordinator(
            (declare_workload(SideReadDefaultWrite()),),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        regions = {
            task_name: region
            for region in compilation.decision.execution_regions
            for task_name in region.tasks
        }
        self.assertEqual(regions["observe"].stream.value, "side_forward")
        self.assertEqual(regions["mutate"].stream.value, "default")
        mutate_advance = next(
            advance
            for advance in compilation.decision.runtime_advance_plan
            if regions["mutate"].region_id in advance.region_ids
        )
        self.assertTrue(mutate_advance.wait_for_prior_side_work)

    def test_join_waits_until_side_host_submission_has_enqueued_work(self) -> None:
        @workload(id="runtime.side-submission-lock")
        class SideSubmissionLock:
            @task(
                outputs={"value": temporary()},
                timing=Between("model.execute", "sampler.pre_draw"),
            )
            def produce(self) -> torch.Tensor:
                return torch.ones(1)

            @task(
                inputs={"value": result("produce", "value")},
                timing=Between("token.commit", "next.prepare"),
            )
            def consume(self, value: torch.Tensor) -> None:
                _ = value.square()

        declaration = declare_workload(SideSubmissionLock())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        side_region = next(
            region
            for region in compilation.decision.execution_regions
            if region.stream.value == "side_forward"
        )
        side_advance = next(
            advance
            for advance in compilation.decision.runtime_advance_plan
            if side_region.region_id in advance.region_ids
        )
        state = ResidualRuntimeState()
        entered = Event()
        release = Event()
        enqueued = Event()
        synchronized = Event()
        wait_observations: list[bool] = []

        class SideStream:
            pass

        class MainStream:
            def wait_stream(self, _stream: SideStream) -> None:
                wait_observations.append(enqueued.is_set())

            def synchronize(self) -> None:
                pass

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=WorkloadExecutionBindings(
                    declarations=(declaration,)
                ),
            )
            execution = state.workload_execution
            self.assertIsNotNone(execution)
            assert execution is not None
            execution.prepare_step(1)
            execution._side_streams = (SideStream(),)
            execution._main_stream = MainStream()

            def submit_nodes(
                _executor: object,
                _step_id: int,
                _nodes: tuple[NodeKey, ...],
                *,
                tensors: tuple[torch.Tensor, ...] = (),
            ) -> object:
                _ = tensors
                entered.set()
                release.wait()
                enqueued.set()
                return object()

            def synchronize_execution() -> None:
                execution.synchronize()
                synchronized.set()

            with mock.patch(
                "tllm.runtime.scheduling.workgraph_executor."
                "PhysicalExecutor.execute_nodes",
                autospec=True,
                side_effect=submit_nodes,
            ), mock.patch.object(
                torch.cuda,
                "stream",
                return_value=contextlib.nullcontext(),
            ):
                submitter = Thread(
                    target=execution.advance_backend_event,
                    args=(side_advance.backend_event,),
                )
                waiter = Thread(
                    target=synchronize_execution
                )
                submitter.start()
                self.assertTrue(entered.wait(1.0))
                waiter.start()
                self.assertFalse(synchronized.wait(0.05))
                release.set()
                submitter.join(1.0)
                waiter.join(1.0)

        self.assertFalse(submitter.is_alive())
        self.assertFalse(waiter.is_alive())
        self.assertTrue(synchronized.is_set())
        self.assertEqual(wait_observations, [True])
        self.assertFalse(execution._side_work_pending)

    def test_physical_region_identity_coverage_fails_closed(self) -> None:
        @workload(id="runtime.identity")
        class Identity:
            @task(
                inputs={"value": provided("value")},
                outputs={"output": sink("answer")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: object) -> object:
                return value

        declaration = declare_workload(Identity())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        physical = compilation.physical
        self.assertIsNotNone(physical)
        assert physical is not None
        original_node = physical.nodes[0]
        unexpected_node = replace(
            original_node,
            node_key=NodeKey("workload_region", "unexpected-region"),
        )
        unexpected_placement = replace(
            physical.anchor_placements[0],
            node_key=unexpected_node.node_key,
        )
        tampered = replace(
            physical,
            nodes=(unexpected_node,),
            anchor_placements=(unexpected_placement,),
        )
        bindings = WorkloadExecutionBindings(
            declarations=(declaration,),
            ports=RuntimePortRegistry(
                inputs={
                    "value": RuntimeInputBinding(
                        lambda _step_id: torch.tensor(1)
                    )
                },
                outputs={
                    "answer": RuntimeOutputBinding(
                        lambda _value, _step_id: None
                    )
                },
            ),
        )

        with (
            mock.patch.object(residual_runtime, "RUNTIME", ResidualRuntimeState()),
            self.assertRaisesRegex(RuntimeError, "identity coverage mismatch"),
        ):
            residual_runtime.publish_workload_compilation(
                replace(compilation, physical=tampered),
                bindings=bindings,
            )

        # PhysicalFlowPlan rejects duplicate nodes at construction. Simulate a
        # corrupt deserialization boundary to keep this runtime guard tested.
        duplicate = replace(physical)
        object.__setattr__(duplicate, "nodes", (original_node, original_node))
        object.__setattr__(
            duplicate,
            "anchor_placements",
            (physical.anchor_placements[0], physical.anchor_placements[0]),
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate physical Region node"):
            _bind_physical_region_nodes(replace(compilation, physical=duplicate))

    def test_physical_region_signed_policy_bindings_fail_closed(self) -> None:
        @workload(id="runtime.signed-policy")
        class SignedPolicy:
            @task(
                inputs={"value": provided("value")},
                outputs={"output": sink("answer")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: object) -> object:
                return value

        declaration = declare_workload(SignedPolicy())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        physical = compilation.physical
        self.assertIsNotNone(physical)
        assert physical is not None
        original_node = physical.nodes[0]

        def assert_rejected(
            tampered_physical: object,
            pattern: str,
        ) -> None:
            with self.assertRaisesRegex(RuntimeError, pattern):
                _bind_physical_region_nodes(
                    replace(compilation, physical=cast(Any, tampered_physical))
                )

        for signature in ("", "0" * 64):
            with self.subTest(schedule_signature=signature or "missing"):
                assert_rejected(
                    replace(physical, schedule_signature=signature),
                    "schedule signature",
                )
        assert_rejected(
            replace(
                physical,
                nodes=(
                    replace(
                        original_node,
                        stream=StreamClass.SIDE_TRAIN,
                    ),
                ),
            ),
            "stream differs",
        )
        assert_rejected(
            replace(
                physical,
                nodes=(
                    replace(
                        original_node,
                        fallback_policy=FallbackPolicy.EAGER,
                    ),
                ),
            ),
            "fallback policy differs",
        )
        artifact_key = original_node.artifact_key
        self.assertIsNotNone(artifact_key)
        assert artifact_key is not None
        assert_rejected(
            replace(
                physical,
                nodes=(
                    replace(
                        original_node,
                        artifact_key=replace(
                            artifact_key,
                            name=f"{artifact_key.name}-tampered",
                        ),
                    ),
                ),
            ),
            "artifact key differs",
        )

        @workload(id="runtime.signed-capture-policy")
        class SignedCapturePolicy:
            @task(
                inputs={"value": hidden(0, "block_output", "decode")},
                outputs={"output": sink("answer")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: object) -> object:
                return value

        timeline = _timeline()
        piecewise = MainWorkloadTimeline(
            backend_id=timeline.backend_id,
            backend_version=timeline.backend_version,
            runner_family=timeline.runner_family,
            graph_mode="piecewise",
            anchors=timeline.anchors,
            regions=timeline.regions,
            static_headroom_bytes=timeline.static_headroom_bytes,
            capabilities=timeline.capabilities,
        )
        capture_compilation = WorkloadCompilerCoordinator(
            (declare_workload(SignedCapturePolicy()),),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="torch_compile",
                residual_resource_manifest=self._indexed_hidden_manifest(
                    (0, "model.layers.0")
                ),
            ),
        ).compile(piecewise)
        capture_physical = capture_compilation.physical
        self.assertIsNotNone(capture_physical)
        assert capture_physical is not None
        self.assertIs(
            capture_physical.nodes[0].capture_policy,
            CapturePolicy.REQUIRE,
        )
        with self.assertRaisesRegex(RuntimeError, "capture policy differs"):
            _bind_physical_region_nodes(
                replace(
                    capture_compilation,
                    physical=replace(
                        capture_physical,
                        nodes=(
                            replace(
                                capture_physical.nodes[0],
                                capture_policy=CapturePolicy.DISABLED,
                            ),
                        ),
                    ),
                )
            )

    def test_piecewise_plan_registers_one_context_graph_per_capture_region(self) -> None:
        contexts: list[object] = []

        class ContextGraph:
            def replay_context(self, context: object) -> None:
                contexts.append(context)

        @workload(id="runtime.capture")
        class Captured:
            @task(
                inputs={"value": hidden(0, "block_output", "decode")},
                outputs={"output": sink("answer")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: object) -> object:
                return cast(torch.Tensor, value) + 1

        declaration = declare_workload(Captured())
        timeline = _timeline()
        piecewise = MainWorkloadTimeline(
            backend_id=timeline.backend_id,
            backend_version=timeline.backend_version,
            runner_family=timeline.runner_family,
            graph_mode="piecewise",
            anchors=timeline.anchors,
            regions=timeline.regions,
            static_headroom_bytes=timeline.static_headroom_bytes,
            capabilities=timeline.capabilities,
        )
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="torch_compile",
                residual_resource_manifest=self._indexed_hidden_manifest(
                    (0, "model.layers.0")
                ),
            ),
        ).compile(piecewise)
        region = compilation.decision.execution_regions[0]
        self.assertTrue(region.capture_safe)
        assert compilation.physical is not None
        self.assertEqual(compilation.physical.nodes[0].capture_policy.value, "require")
        state = ResidualRuntimeState()
        graph = ContextGraph()

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch("torch.compile", side_effect=lambda function, **_kwargs: function),
            mock.patch(
                "tllm.runtime.workload_execution._RegionCudaGraph",
                return_value=graph,
            ) as graph_factory,
        ):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=WorkloadExecutionBindings(
                    declarations=(declaration,),
                    ports=RuntimePortRegistry(
                        outputs={
                            "answer": RuntimeOutputBinding(
                                lambda _value, _step_id: None
                            )
                        }
                    ),
                    residual_paths={
                        ResidualLocator(0, "block_output", "decode"): "model.layers.0"
                    },
                ),
            )
            execution = state.workload_execution
            assert execution is not None
            residual_runtime.prepare_physical_workgraph_step(1)
            residual_runtime.advance_workload_anchor(region.placement_anchor)

        self.assertEqual(graph_factory.call_count, 1)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(cast(Any, contexts[0]).kind.value, "replay")
        self.assertEqual(execution.executor.stats.replay_count, 1)
        self.assertEqual(execution.executor.stats.eager_count, 0)

    def test_missing_runtime_value_or_sink_binding_fails_before_execution(self) -> None:
        @workload(id="runtime.bindings")
        class Bound:
            @task(
                inputs={"value": provided("value")},
                outputs={"output": sink("answer")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: object) -> object:
                return value

        declaration = declare_workload(Bound())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())

        with self.assertRaisesRegex(RuntimeError, "provided input.*value"):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=WorkloadExecutionBindings(
                    declarations=(declaration,),
                    ports=RuntimePortRegistry(
                        outputs={
                            "answer": RuntimeOutputBinding(
                                lambda _value, _step_id: None
                            )
                        }
                    ),
                ),
            )
        with self.assertRaisesRegex(RuntimeError, "output sink.*answer"):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=WorkloadExecutionBindings(
                    declarations=(declaration,),
                    ports=RuntimePortRegistry(
                        inputs={
                            "value": RuntimeInputBinding(
                                lambda _step_id: 1
                            )
                        }
                    ),
                ),
            )

    def test_provided_step_id_name_is_not_the_builtin_step_input(self) -> None:
        @workload(id="runtime.typed-step-binding")
        class TypedStepBinding:
            @task(
                inputs={
                    "builtin_step": step_id(),
                    "user_step": provided("step_id"),
                },
                timing=Between("model.execute", "step.end"),
            )
            def run(self, builtin_step: int, user_step: object) -> None:
                _ = builtin_step, user_step

        declaration = declare_workload(TypedStepBinding())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())

        with self.assertRaisesRegex(RuntimeError, "provided input.*step_id"):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=WorkloadExecutionBindings(declarations=(declaration,)),
            )

        state = ResidualRuntimeState()
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=WorkloadExecutionBindings(
                    declarations=(declaration,),
                    ports=RuntimePortRegistry(
                        inputs={
                            "step_id": RuntimeInputBinding(
                                lambda _step_id: "user-value"
                            )
                        }
                    ),
                ),
            )
            execution = state.workload_execution
            assert execution is not None
            execution.close()

    def test_hidden_resource_bindings_fail_closed_during_install(self) -> None:
        source_locator = ResidualLocator(0, "block_output", "decode")
        target_locator = ResidualLocator(-1, "block_output", "decode")

        @workload(id="runtime.hidden-resource-index")
        class Indexed:
            @task(
                inputs={
                    "source": hidden(0, "block_output", "decode"),
                    "target": hidden(-1, "block_output", "decode"),
                },
                outputs={"output": sink("answer")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, source: object, target: object) -> object:
                _ = target
                return source

        indexed = declare_workload(Indexed())
        indexed_compilation = WorkloadCompilerCoordinator(
            (indexed,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
                residual_resource_manifest=self._indexed_hidden_manifest(
                    (-1, "model.layers.-1"),
                    (0, "model.layers.0"),
                ),
            ),
        ).compile(_timeline())

        def install(
            compilation: object,
            declaration: object,
            residual_paths: dict[ResidualLocator, str],
        ) -> ResidualRuntimeState:
            state = ResidualRuntimeState()
            with mock.patch.object(residual_runtime, "RUNTIME", state):
                residual_runtime.publish_workload_compilation(
                    cast(Any, compilation),
                    bindings=WorkloadExecutionBindings(
                        declarations=(cast(Any, declaration),),
                        ports=RuntimePortRegistry(
                            outputs={
                                "answer": RuntimeOutputBinding(
                                    lambda _value, _step_id: None
                                )
                            }
                        ),
                        residual_paths=cast("Mapping[object, str]", residual_paths),
                    ),
                )
            return state

        state = install(
            indexed_compilation,
            indexed,
            {
                source_locator: "model.layers.0",
                target_locator: "model.layers.-1",
            },
        )
        self.assertIsNotNone(state.workload_execution)

        with self.assertRaisesRegex(RuntimeError, "residual resource manifest coverage mismatch"):
            install(
                indexed_compilation,
                indexed,
                {source_locator: "model.layers.0"},
            )
        with self.assertRaisesRegex(RuntimeError, "unexpected residual resource"):
            install(
                indexed_compilation,
                indexed,
                {
                    source_locator: "model.layers.0",
                    target_locator: "model.layers.-1",
                    ResidualLocator(1, "block_output", "decode"): "model.layers.1",
                },
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "residual resource manifest does not match",
        ):
            install(
                indexed_compilation,
                indexed,
                {
                    source_locator: "model.layers.shared",
                    target_locator: "model.layers.shared",
                },
            )

        @workload(id="runtime.hidden-resource-path")
        class Path:
            @task(
                inputs={"value": hidden("model.layers.0", "block_output", "decode")},
                outputs={"output": sink("answer")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: object) -> object:
                return value

        path = declare_workload(Path())
        path_compilation = WorkloadCompilerCoordinator(
            (path,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        with self.assertRaisesRegex(RuntimeError, "missing PATH hidden resource"):
            install(path_compilation, path, {})
        state = install(
            path_compilation,
            path,
            {source_locator: "model.layers.0"},
        )
        self.assertIsNotNone(state.workload_execution)

        @workload(id="runtime.hidden-resource-wildcard")
        class Wildcard:
            @task(
                inputs={"value": hidden("*", "block_output", "decode")},
                outputs={"output": sink("answer")},
                timing=Between("model.execute", "step.end"),
            )
            def run(self, value: object) -> object:
                return value

        wildcard = declare_workload(Wildcard())
        wildcard_compilation = WorkloadCompilerCoordinator(
            (wildcard,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        with self.assertRaisesRegex(RuntimeError, "wildcard hidden input"):
            install(wildcard_compilation, wildcard, {})

    def test_signed_state_effect_without_version_registry_fails_closed(self) -> None:
        @workload(id="runtime.state")
        class Stateful:
            state = State(scope=StateScope.REPLICA, parameters="model")

            @task(
                effects=(read(state),),
                timing=Between("model.execute", "step.end"),
            )
            def run(self) -> None:
                return None

        declaration = declare_workload(Stateful())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        self.assertEqual(compilation.graph.states[0].resource, "state")
        self.assertEqual(
            compilation.decision.execution_regions[0].effects[0].resource,
            "state",
        )

        with self.assertRaisesRegex(RuntimeError, "state/version registry"):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=WorkloadExecutionBindings(declarations=(declaration,)),
            )

    def test_signed_state_transition_advances_bound_version_once_per_region(self) -> None:
        versions: dict[Hashable, int] = {None: 0}

        @workload(id="runtime.state.transition")
        class Stateful:
            state = State(
                scope=StateScope.REPLICA,
                parameters="model",
                transition="manual",
            )

            @task(
                effects=(transition(state),),
                timing=Between("model.execute", "step.end"),
            )
            def update(self) -> None:
                versions[None] += 1

        declaration = declare_workload(Stateful())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        schedule_state = compilation.graph.states[0]
        state = ResidualRuntimeState()
        bindings = WorkloadExecutionBindings(
            declarations=(declaration,),
            state_versions={
                schedule_state.resource: StateVersionBinding(
                    schedule_state,
                    lambda _step_id: dict(versions),
                )
            },
        )

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=bindings,
            )
            execution = state.workload_execution
            self.assertIsNotNone(execution)
            assert execution is not None
            anchor = compilation.decision.execution_regions[0].placement_anchor
            residual_runtime.prepare_physical_workgraph_step(1)
            self.assertTrue(residual_runtime.advance_workload_anchor(anchor))
            residual_runtime.prepare_physical_workgraph_step(2)
            self.assertTrue(residual_runtime.advance_workload_anchor(anchor))

        self.assertEqual(versions[None], 2)
        assert execution.state_versions is not None
        self.assertEqual(
            execution.state_versions.observed_version(schedule_state.resource, None),
            2,
        )

    def test_signed_state_transition_without_version_publish_fails(self) -> None:
        versions: dict[Hashable, int] = {None: 3}

        @workload(id="runtime.state.invalid-transition")
        class Stateful:
            state = State(
                scope=StateScope.REPLICA,
                parameters="model",
                transition="manual",
            )

            @task(
                effects=(transition(state),),
                timing=Between("model.execute", "step.end"),
            )
            def update(self) -> None:
                return None

        declaration = declare_workload(Stateful())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        schedule_state = compilation.graph.states[0]
        state = ResidualRuntimeState()
        bindings = WorkloadExecutionBindings(
            declarations=(declaration,),
            state_versions={
                schedule_state.resource: StateVersionBinding(
                    schedule_state,
                    lambda _step_id: dict(versions),
                )
            },
        )

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=bindings,
            )
            residual_runtime.prepare_physical_workgraph_step(1)
            anchor = compilation.decision.execution_regions[0].placement_anchor
            with self.assertRaisesRegex(RuntimeError, "signed mutation version effect"):
                residual_runtime.advance_workload_anchor(anchor)

    def _run_conditional_transition(self, bump: int) -> None:
        versions: dict[Hashable, int] = {None: 0}

        @workload(id=f"runtime.state.maybe.{bump}")
        class Stateful:
            state = State(
                scope=StateScope.REPLICA,
                parameters="model",
                transition="manual",
            )

            @task(
                effects=(maybe_transition(state),),
                timing=Between("model.execute", "step.end"),
            )
            def update(self) -> None:
                versions[None] += bump

        declaration = declare_workload(Stateful())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        schedule_state = compilation.graph.states[0]
        state = ResidualRuntimeState()
        bindings = WorkloadExecutionBindings(
            declarations=(declaration,),
            state_versions={
                schedule_state.resource: StateVersionBinding(
                    schedule_state,
                    lambda _step_id: dict(versions),
                )
            },
        )
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=bindings,
            )
            residual_runtime.prepare_physical_workgraph_step(1)
            anchor = compilation.decision.execution_regions[0].placement_anchor
            residual_runtime.advance_workload_anchor(anchor)

    def test_signed_conditional_transition_accepts_skip_or_fire(self) -> None:
        # The audit accepts exactly +0 (data-dependent skip) or +1 (fired).
        self._run_conditional_transition(0)
        self._run_conditional_transition(1)

    def test_signed_conditional_transition_rejects_any_other_delta(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "signed mutation version effect"):
            self._run_conditional_transition(2)

    def _run_double_conditional_transition(self, bump_a: int, bump_b: int) -> None:
        versions: dict[Hashable, int] = {None: 0}

        @workload(id=f"runtime.state.maybe2.{bump_a}.{bump_b}")
        class Stateful:
            state = State(
                scope=StateScope.REPLICA,
                parameters="model",
                transition="manual",
            )

            @task(
                effects=(maybe_transition(state),),
                timing=Between("model.execute", "step.end"),
            )
            def update_a(self) -> None:
                versions[None] += bump_a

            @task(
                effects=(maybe_transition(state),),
                timing=Between("model.execute", "step.end"),
            )
            def update_b(self) -> None:
                versions[None] += bump_b

        declaration = declare_workload(Stateful())
        compilation = WorkloadCompilerCoordinator(
            (declaration,),
            options=CompileOptions(
                memory_budget_bytes=1 << 20,
                compile_policy="eager",
            ),
        ).compile(_timeline())
        regions = compilation.decision.execution_regions
        # Both conditional transitions land in ONE Region (same window,
        # internalized effects) — the case the interval audit must cover.
        self.assertEqual(len(regions), 1)
        schedule_state = compilation.graph.states[0]
        state = ResidualRuntimeState()
        bindings = WorkloadExecutionBindings(
            declarations=(declaration,),
            state_versions={
                schedule_state.resource: StateVersionBinding(
                    schedule_state,
                    lambda _step_id: dict(versions),
                )
            },
        )
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=bindings,
            )
            residual_runtime.prepare_physical_workgraph_step(1)
            anchor = regions[0].placement_anchor
            residual_runtime.advance_workload_anchor(anchor)

    def test_two_conditional_transitions_accept_any_partial_fire(self) -> None:
        # Any subset of the signed conditional transitions may fire:
        # +0, +1 (either one) and +2 are all legal.
        self._run_double_conditional_transition(0, 0)
        self._run_double_conditional_transition(1, 0)
        self._run_double_conditional_transition(0, 1)
        self._run_double_conditional_transition(1, 1)

    def test_two_conditional_transitions_reject_beyond_signed_total(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "signed mutation version effect"):
            self._run_double_conditional_transition(1, 2)

    def test_dummy_compilation_takes_authority_at_decision_anchor(self) -> None:
        workload = DummyWorkload(
            DummyConfig(
                enable_async=False,
                export_every_n_steps=2,
                export_max_rows=4,
                export_max_cols=8,
            )
        )
        runtime = TllmRuntime(
            workloads=(workload,),
            workload_memory_budget_bytes=1 << 20,
        )
        compilation = runtime.compile_workloads(_timeline())
        state = ResidualRuntimeState()
        locator = ResidualLocator(layer=0, site="block_output", phase="decode")
        hidden = torch.randn(3, 5)
        bindings = runtime.workload_execution_bindings
        self.assertIsNotNone(bindings)
        assert bindings is not None

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            state.residual_bindings = {
                "model.layers.0": ResidualPathBinding(locator, "model.layers.0")
            }
            state.tap_decode_hidden["model.layers.0"] = hidden
            state.decode_count = 3
            state.event_step_id = 2
            residual_runtime.publish_workload_compilation(compilation, bindings=bindings)

            execution = state.workload_execution
            self.assertIsNotNone(execution)
            assert execution is not None
            self.assertEqual(runtime.consumers, ())
            self.assertEqual(execution.completion_anchor, "token.commit")

            residual_runtime.prepare_physical_workgraph_step(2)
            self.assertTrue(residual_runtime.advance_workload_anchor("token.commit"))

        stats = workload.exporter.diagnostics()
        self.assertEqual(stats["consumed_batches"], 1.0)
        self.assertEqual(stats["consumed_rows"], 3.0)
        self.assertEqual(stats["processed_batches"], 1.0)
        self.assertEqual(execution.executor.stats.execute_count, 1)
        self.assertEqual(execution.executor.stats.eager_count, 1)

    def test_dummy_native_handler_preserves_dispatch_stride(self) -> None:
        workload = DummyWorkload(
            DummyConfig(
                enable_async=False,
                export_every_n_steps=2,
                export_max_rows=4,
            )
        )
        runtime = TllmRuntime(
            workloads=(workload,),
            workload_memory_budget_bytes=1 << 20,
        )
        state = ResidualRuntimeState()
        locator = ResidualLocator(layer=0, site="block_output", phase="decode")
        bindings = runtime.workload_execution_bindings
        self.assertIsNotNone(bindings)
        assert bindings is not None

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            state.residual_bindings = {
                "model.layers.0": ResidualPathBinding(locator, "model.layers.0")
            }
            state.tap_decode_hidden["model.layers.0"] = torch.randn(2, 4)
            state.decode_count = 2
            state.event_step_id = 1
            residual_runtime.publish_workload_compilation(
                runtime.compile_workloads(_timeline()), bindings=bindings
            )
            residual_runtime.prepare_physical_workgraph_step(1)
            residual_runtime.advance_workload_anchor("token.commit")

        stats = workload.exporter.diagnostics()
        self.assertEqual(stats["consumed_batches"], 0.0)
        self.assertEqual(stats["processed_batches"], 0.0)
        assert state.workload_execution is not None
        self.assertEqual(state.workload_execution.executor.stats.execute_count, 1)

    def test_engine_warmup_anchor_before_first_prepare_is_ignored(self) -> None:
        workload = DummyWorkload()
        runtime = TllmRuntime(
            workloads=(workload,),
            workload_memory_budget_bytes=1 << 20,
        )
        state = ResidualRuntimeState()
        bindings = runtime.workload_execution_bindings
        self.assertIsNotNone(bindings)
        assert bindings is not None

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                runtime.compile_workloads(_timeline()), bindings=bindings
            )
            execution = state.workload_execution
            self.assertIsNotNone(execution)
            assert execution is not None

            self.assertFalse(execution.advance_anchor("token.commit"))

        self.assertEqual(execution.executor.stats.prepare_count, 0)
        self.assertEqual(execution.executor.stats.execute_count, 0)

    def test_native_dummy_plan_without_bindings_fails_closed(self) -> None:
        workload = DummyWorkload()
        runtime = TllmRuntime(
            workloads=(workload,),
            workload_memory_budget_bytes=1 << 20,
        )
        state = ResidualRuntimeState()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            state.native_workloads_pending = True
            with self.assertRaisesRegex(RuntimeError, "requires explicit bindings"):
                residual_runtime.publish_workload_compilation(
                    runtime.compile_workloads(_timeline())
                )

    def test_native_dummy_rejects_legacy_only_feedback_and_capture_policies(self) -> None:
        with self.assertRaisesRegex(ValueError, "declared fields"):
            DummyWorkload(DummyConfig(feedback_interval=1))
        with self.assertRaisesRegex(ValueError, "declared fields"):
            DummyWorkload(DummyConfig(capture_policy="required"))

    def test_dummy_region_rejects_a_backend_without_its_eager_capability(self) -> None:
        workload = DummyWorkload()
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        eager = _timeline()
        piecewise = MainWorkloadTimeline(
            backend_id=eager.backend_id,
            backend_version=eager.backend_version,
            runner_family=eager.runner_family,
            graph_mode="piecewise",
            anchors=eager.anchors,
            regions=eager.regions,
            static_headroom_bytes=eager.static_headroom_bytes,
            capabilities=tuple(
                capability
                for capability in eager.capabilities
                if capability != "eager_only"
            ),
        )

        with self.assertRaisesRegex(Exception, "eager_only"):
            runtime.compile_workloads(piecewise)

    def test_dummy_noise_configuration_isolated_in_cache_and_artifact_keys(self) -> None:
        quiet = TllmRuntime(
            workloads=(DummyWorkload(DummyConfig(noise_std=0.0)),),
            workload_memory_budget_bytes=1 << 20,
        )
        noisy = TllmRuntime(
            workloads=(DummyWorkload(DummyConfig(noise_std=0.25)),),
            workload_memory_budget_bytes=1 << 20,
        )

        quiet_compilation = quiet.compile_workloads(_timeline())
        noisy_compilation = noisy.compile_workloads(_timeline())

        self.assertNotEqual(quiet_compilation.cache_key, noisy_compilation.cache_key)
        self.assertNotEqual(
            quiet_compilation.decision.execution_regions[0].artifact_key,
            noisy_compilation.decision.execution_regions[0].artifact_key,
        )

    def test_dummy_region_requires_a_resolved_native_residual_binding(self) -> None:
        workload = DummyWorkload()
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        compilation = runtime.compile_workloads(_timeline())
        bindings = runtime.workload_execution_bindings
        self.assertIsNotNone(bindings)
        assert bindings is not None
        state = ResidualRuntimeState()
        state.decode_count = 1

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(compilation, bindings=bindings)
            assert state.workload_execution is not None
            self.assertTrue(state.workload_execution.requires_decode_hidden)
            residual_runtime.prepare_physical_workgraph_step(1)
            with self.assertRaisesRegex(RuntimeError, "no resolved hidden input"):
                residual_runtime.advance_workload_anchor("token.commit")

    def test_dummy_worker_reconstruction_preserves_resources_and_signed_graph(self) -> None:
        workload = DummyWorkload(
            DummyConfig(
                enable_async=False,
                export_every_n_steps=3,
                export_max_rows=2,
                export_max_cols=7,
            )
        )
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        compilation = runtime.compile_workloads(_timeline())
        resources = workload.native_workload_resources()
        self.assertIsInstance(resources.worker, WorkerRuntimeBinding)
        assert isinstance(resources.worker, WorkerRuntimeBinding)
        worker_spec = resources.worker.spec

        bindings = reconstruct_worker_bindings(compilation, (worker_spec,))

        self.assertEqual(
            bindings.declarations[0].ir.semantic_key,
            declare_workload(workload).ir.semantic_key,
        )
        self.assertEqual(bindings.ports.inputs, {})
        self.assertEqual(bindings.ports.outputs, {})
        self.assertEqual(
            bindings.residual_paths,
            {ResidualLocator(0, "block_output", "decode"): "model.layers.0"},
        )
        self.assertIs(type(bindings.lifecycle), ExecutionLifecycleBinding)

    def test_dummy_worker_reconstruction_sends_signed_facts_without_specializer(self) -> None:
        workload = DummyWorkload()
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        specialization = RuntimeSpecializationSpec(
            2, 4, torch.float32, torch.device("cpu")
        )
        # Dummy has no cold specializer, but the parent still signs the runner
        # specialization into the Region artifact identity.
        runtime.cold_specialize_workloads(specialization)
        compilation = runtime.compile_workloads(_timeline())
        worker = workload.native_workload_resources().worker
        assert isinstance(worker, WorkerRuntimeBinding)
        worker_spec = worker.spec

        bindings = reconstruct_worker_bindings(
            compilation, (worker_spec,), specialization=specialization
        )

        self.assertEqual(bindings.declarations[0].ir.workload_id, "dummy")

    def test_in_process_cold_specialization_uses_rust_identity_for_reuse(self) -> None:
        from tllm import _core

        runtime = TllmRuntime(
            workloads=(DummyWorkload(),), workload_memory_budget_bytes=1 << 20
        )
        initial_compiler = runtime._workload_compiler
        specialization = RuntimeSpecializationSpec(
            2, 4, torch.float32, torch.device("cpu"), model_layer_count=3
        )

        with (
            mock.patch.object(
                _core,
                "normalize_runtime_specialization_options",
                wraps=_core.normalize_runtime_specialization_options,
            ) as normalize,
            mock.patch.object(
                runtime,
                "_rebuild_workload_compiler",
                wraps=runtime._rebuild_workload_compiler,
            ) as rebuild,
        ):
            runtime.cold_specialize_workloads(specialization)
            specialized_compiler = runtime._workload_compiler
            runtime.cold_specialize_workloads(specialization)

        self.assertEqual(normalize.call_count, 2)
        self.assertEqual(rebuild.call_count, 1)
        self.assertIsNot(specialized_compiler, initial_compiler)
        self.assertIs(runtime._workload_compiler, specialized_compiler)
        self.assertIn(
            ("runtime/model_layer_count", "3"),
            runtime.workload_compile_options.shape_specialization,
        )
        with self.assertRaisesRegex(RuntimeError, "specialization does not match"):
            runtime.cold_specialize_workloads(
                RuntimeSpecializationSpec(
                    3, 4, torch.float32, torch.device("cpu"), model_layer_count=3
                )
            )
        self.assertIs(runtime._workload_compiler, specialized_compiler)

    def test_in_process_cold_specialization_rejects_invalid_facts_before_binding(self) -> None:
        runtime = TllmRuntime(
            workloads=(DummyWorkload(),), workload_memory_budget_bytes=1 << 20
        )
        initial_options = runtime.workload_compile_options
        initial_compiler = runtime._workload_compiler
        # Bypass the Python constructor to exercise the Rust FFI fail-closed
        # path used if an invalid typed object reaches this boundary.
        invalid = object.__new__(RuntimeSpecializationSpec)
        object.__setattr__(invalid, "rows", 0)
        object.__setattr__(invalid, "hidden_size", 4)
        object.__setattr__(invalid, "dtype", torch.float32)
        object.__setattr__(invalid, "device", torch.device("cpu"))
        object.__setattr__(invalid, "model_layer_count", None)
        object.__setattr__(invalid, "max_prefill_tokens", None)

        with self.assertRaisesRegex(ValueError, "runtime specialization facts"):
            runtime.cold_specialize_workloads(invalid)

        self.assertIs(runtime.workload_compile_options, initial_options)
        self.assertIs(runtime._workload_compiler, initial_compiler)

    def test_cold_specialization_rejects_malformed_static_manifest_before_callbacks(
        self,
    ) -> None:
        workload = DummyWorkload()
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        cold_specializer = mock.Mock()
        resources = runtime._native_workload_resources[id(workload)]
        assert isinstance(resources.services, LocalRuntimeServices)
        runtime._native_workload_resources[id(workload)] = replace(
            resources,
            services=replace(resources.services, cold_specialize=cold_specializer),
        )
        runtime._workload_compile_options = replace(
            runtime.workload_compile_options,
            static_binding_manifest=(
                '[{"workload_id":"dummy","factory":"factory","config":[],'
                '"specialization_key":[],"backend_region_implementation_ids":[]}]'
            ),
        )
        initial_options = runtime.workload_compile_options
        initial_compiler = runtime._workload_compiler
        specialization = RuntimeSpecializationSpec(
            2, 4, torch.float32, torch.device("cpu")
        )

        with self.assertRaisesRegex(ValueError, "config must be a JSON object"):
            runtime.cold_specialize_workloads(specialization)

        cold_specializer.assert_not_called()
        self.assertIs(runtime.workload_compile_options, initial_options)
        self.assertIs(runtime._workload_compiler, initial_compiler)

    def test_reconstructed_dummy_diagnostics_describe_the_active_resources(self) -> None:
        workload = DummyWorkload(
            DummyConfig(enable_async=False, export_every_n_steps=1)
        )
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        compilation = runtime.compile_workloads(_timeline())
        worker = workload.native_workload_resources().worker
        assert isinstance(worker, WorkerRuntimeBinding)
        worker_spec = worker.spec
        bindings = reconstruct_worker_bindings(compilation, (worker_spec,))
        state = ResidualRuntimeState()
        locator = ResidualLocator(layer=0, site="block_output", phase="decode")

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            state.residual_bindings = {
                "model.layers.0": ResidualPathBinding(locator, "model.layers.0")
            }
            state.tap_decode_hidden["model.layers.0"] = torch.randn(2, 4)
            state.decode_count = 2
            state.event_step_id = 1
            residual_runtime.publish_workload_compilation(compilation, bindings=bindings)
            residual_runtime.prepare_physical_workgraph_step(1)
            self.assertTrue(residual_runtime.advance_workload_anchor("token.commit"))
            assert state.workload_execution is not None
            active_diagnostics = state.workload_execution.read_resource_diagnostics()

        self.assertGreater(active_diagnostics["dummy"]["consumed_batches"], 0)
        self.assertGreater(active_diagnostics["dummy"]["processed_batches"], 0)
        self.assertEqual(workload.exporter.diagnostics()["consumed_batches"], 0.0)

    def test_resource_diagnostics_are_namespaced_and_fail_closed(self) -> None:
        execution = NativeWorkloadExecution(
            compilation=cast(Any, object()),
            executor=cast(Any, object()),
            completion_anchor="step.end",
            _owned_consumers=(),
            _diagnostics_readers={
                "alpha": lambda: {"count": 2, "latency": 0.5},
                "beta": lambda: {"count": 3},
            },
        )
        self.assertEqual(
            execution.read_resource_diagnostics(),
            {
                "alpha": {"count": 2, "latency": 0.5},
                "beta": {"count": 3},
            },
        )

        for invalid in (
            {"": 1},
            {"count": True},
            {"count": float("nan")},
            {"count": "1"},
        ):
            invalid_execution = NativeWorkloadExecution(
                compilation=cast(Any, object()),
                executor=cast(Any, object()),
                completion_anchor="step.end",
                _owned_consumers=(),
                _diagnostics_readers={"invalid": lambda value=invalid: value},
            )
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_execution.read_resource_diagnostics()


if __name__ == "__main__":
    unittest.main()

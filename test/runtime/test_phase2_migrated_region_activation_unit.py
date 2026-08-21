"""Red contracts for activating a migrated WorkGraph capture region."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety, SamplerStage
from tllm.consumers.base import BaseConsumer
from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.contracts.runtime_context import RuntimeContext
from tllm.producers.registry import ExecutionPlan
from tllm.runtime import residual_runtime
from tllm.runtime.vllm_patch import port_runtime_hooks
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.execution_plan import compile_default_plan
from tllm.runtime.graph_policy import GraphMode
from tllm.runtime.scheduling.physical import Fence
from tllm.runtime.scheduling.plan_store import (
    PlanResources,
    PlanSnapshotStore,
    RuntimeStream,
)
from tllm.runtime.scheduling.workgraph_executor import (
    PhysicalExecutionContext,
    PhysicalExecutionError,
    PhysicalExecutionKind,
    PhysicalExecutor,
)
from tllm.runtime.scheduling.phase3_transfer import PlacementRuntimeBinding
from tllm.runtime.state import ResidualRuntimeState
from tllm.workgraph import placement
from tllm.workgraph import (
    ArtifactKey,
    Capability,
    CaptureAttachment,
    CaptureStatus,
    CapturedFlow,
    Device,
    Effect,
    ExecutorKey,
    NodeKey,
    StateScope,
    compile_shadow_parity,
    flow,
)
from tllm.workgraph.frontend import ExternContract
from tllm.workgraph.anchors import BackendResolution, shadow_backend_snapshot
from tllm.workgraph import physical as physical_module
from tllm.workgraph.shadow import ShadowParityPlan
from tllm.workgraph.lowering import WorkGraphError
from tllm.workgraph.physical import (
    CapturePolicy,
    FallbackPolicy,
    PhysicalCaptureRegion,
    PhysicalFlowPlan,
    FenceKey,
    StreamClass,
)


class _FakeStream:
    def wait_event(self, event: object) -> None:
        del event


class _FakeEvent:
    def __init__(self) -> None:
        self.done = True

    def record(self, stream: object) -> None:
        del stream

    def query(self) -> bool:
        return self.done


def _runtime_stream() -> RuntimeStream:
    return RuntimeStream(
        StreamClass.SIDE_FORWARD,
        _FakeStream(),
        Fence(
            FenceKey("phase2-region", "retire-side-forward"),
            _FakeEvent(),
        ),
    )


class _RecordingESampConsumer(ESampConsumer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.activated_regions: list[object] = []

    def activate_workgraph_regions(self, regions: tuple[object, ...]) -> None:
        self.activated_regions.extend(regions)


class _PlacementRecordingESampConsumer(ESampConsumer):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.activated_resource_batches: list[tuple[PlanResources, ...]] = []

    def activate_workgraph_region_resources(
        self,
        regions: tuple[PhysicalCaptureRegion, ...],
        resources: tuple[PlanResources, ...],
    ) -> None:
        self.activated_resource_batches.append(resources)
        super().activate_workgraph_region_resources(regions, resources)


class _RecordingRuntimeImplementation:
    def __init__(self, label: str, trace: list[tuple[str, tuple[object, ...]]]) -> None:
        self._label = label
        self._trace = trace

    def execute(self, *args: object) -> None:
        self._trace.append((self._label, args))


def _resolution() -> BackendResolution:
    return BackendResolution(
        capabilities=BackendCapabilities(
            backend_id="vllm-phase2-region",
            runner_family="mrv2",
            sampler_stages=frozenset({SamplerStage.BEFORE_DRAW}),
            native_min_p=True,
            tuple_sampler_return=True,
            split_execute_and_sample=True,
            graph_safety=GraphSafety.CAPTURE_SAFE,
        ),
        graph_mode=GraphMode.PIECEWISE,
    )


def _esamp_phase1(
    consumer: ESampConsumer | None = None,
) -> tuple[ESampConsumer, object, object]:
    actual = consumer or ESampConsumer(ESampConsumerConfig(graph_scratch_rows=8))
    captured, attachment = actual.specialize_workgraph(
        torch.ones(2, 4),
        torch.zeros(2, 4),
    )
    dispatch_plan = DispatchPlan.build([actual])
    execution_plan = compile_default_plan(consumers=[actual])
    phase1 = compile_shadow_parity(
        consumers=(actual,),
        dispatch_plan=dispatch_plan,
        execution_plan=execution_plan,
        backend=shadow_backend_snapshot(),
        captures=(captured,),
        attachments=(attachment,),
    )
    return actual, phase1, captured


def _compile_region(
    test: unittest.TestCase,
    phase1: object,
    artifact_key: ArtifactKey,
) -> object:
    compiler = getattr(physical_module, "compile_physical_capture_region", None)
    test.assertTrue(
        callable(compiler),
        "Phase 2 must expose compile_physical_capture_region(phase1, artifact_key)",
    )
    assert callable(compiler)
    region = compiler(phase1, artifact_key)
    region_type = getattr(physical_module, "PhysicalCaptureRegion", None)
    test.assertIsNotNone(
        region_type,
        "Phase 2 must expose the frozen PhysicalCaptureRegion descriptor",
    )
    assert region_type is not None
    test.assertIsInstance(region, region_type)
    return region


def _region_plan(test: unittest.TestCase, region: object) -> PhysicalFlowPlan:
    plan = getattr(region, "plan", None)
    test.assertIsInstance(plan, PhysicalFlowPlan)
    assert isinstance(plan, PhysicalFlowPlan)
    return plan


def _bundle(source: torch.Tensor, target: torch.Tensor) -> PortBundle:
    return PortBundle(
        key=BundleKey(
            engine_step_id=1,
            phase="decode",
            request_id="req-a",
            sample_idx=0,
        ),
        entries={
            "device_lease": DeviceTensorLease(
                entries={"source": source, "target": target},
                active_rows=int(source.shape[0]),
            ),
            "request_meta": RowBatchMeta(
                request_ids=("req-a", "req-b"),
                prompt_idxs=(0, 1),
                sample_idxs=(0, 0),
                phase="decode",
                engine_step_id=1,
            ),
        },
    )


def _context() -> RuntimeContext:
    return RuntimeContext(
        runner=SimpleNamespace(max_num_reqs=8),
        model=None,
        device=torch.device("cpu"),
        main_stream=None,
        is_compiling=False,
        uses_cudagraph=False,
        event_name="flow:decode",
    )


def _step_bundle(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    step_id: int,
) -> PortBundle:
    return PortBundle(
        key=BundleKey(
            engine_step_id=step_id,
            phase="decode",
            request_id="req-a",
            sample_idx=0,
        ),
        entries={
            "device_lease": DeviceTensorLease(
                entries={"source": source, "target": target},
                active_rows=int(source.shape[0]),
            ),
            "request_meta": RowBatchMeta(
                request_ids=("req-a", "req-b"),
                prompt_idxs=(0, 1),
                sample_idxs=(0, 0),
                phase="decode",
                engine_step_id=step_id,
            ),
        },
    )


class Phase2MigratedRegionActivationUnitTest(unittest.TestCase):
    def test_base_consumer_fails_closed_on_unhandled_placement_resources(self) -> None:
        class UnsupportedConsumer(BaseConsumer[object]):
            config = object()

            @property
            def consumer_id(self) -> str:
                return "unsupported-placement-consumer"

        resources = PlanResources(plan=PhysicalFlowPlan(plan_epoch=1, nodes=()))

        with self.assertRaisesRegex(
            RuntimeError,
            "does not support WorkGraph placement resources",
        ):
            UnsupportedConsumer().activate_workgraph_region_resources((), (resources,))

    def test_residual_activation_executes_selected_esamp_region_implementation(self) -> None:
        engine = mock.Mock()
        consumer = _PlacementRecordingESampConsumer(
            ESampConsumerConfig(graph_scratch_rows=8),
            engine=engine,
        )
        _consumer, phase1, _captured = _esamp_phase1(consumer)
        self.assertIsInstance(phase1, ShadowParityPlan)
        assert isinstance(phase1, ShadowParityPlan)
        root_plan = physical_module.compile_physical_flow_plan(phase1)
        region_executor = consumer.workgraph_region_executors()[0].executor
        trace: list[tuple[str, tuple[object, ...]]] = []
        registry = placement.ImplementationRegistry()
        logical_executors = tuple(
            dict.fromkeys((*tuple(node.executor for node in root_plan.nodes), region_executor))
        )
        selected_key: ExecutorKey | None = None
        for ordinal, logical_executor in enumerate(logical_executors):
            is_region = logical_executor == region_executor
            implementation_key = ExecutorKey(
                "implementation",
                f"placement.esamp.region-{ordinal}",
            )
            registry.register(
                placement.ImplementationSpec(
                    key=implementation_key,
                    logical_executor=logical_executor,
                    placement=placement.Placement(placement.DeviceKind.CPU),
                    implementation=_RecordingRuntimeImplementation(
                        "selected" if is_region else f"root-{ordinal}",
                        trace,
                    ),
                )
            )
            if is_region:
                selected_key = implementation_key
        self.assertIsNotNone(selected_key)
        binding = PlacementRuntimeBinding(
            registry=registry,
            topology=placement.DeviceTopology(peer_access=frozenset()),
        )
        state = ResidualRuntimeState()
        source = torch.randn(2, 4)
        target = torch.randn(2, 4)
        legacy_staging = mock.Mock(
            side_effect=AssertionError("legacy ESamp staging handler executed")
        )

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch.object(
                consumer,
                "_execute_workgraph_staging",
                legacy_staging,
            ),
        ):
            residual_runtime.configure_workgraph_placement(binding)
            residual_runtime.register_dispatch_consumer(consumer)
            residual_runtime.record_workgraph_backend_resolution(
                resolution=_resolution(),
            )
            residual_runtime.finalize_workgraph_snapshot_after_placement_refresh()

            self.assertEqual(len(consumer.activated_resource_batches), 1)
            (activation_resources,) = consumer.activated_resource_batches[0]
            placement_plan = activation_resources.placement_plan
            self.assertIsNotNone(placement_plan)
            assert placement_plan is not None
            self.assertEqual(
                placement_plan.selections[0].key,
                selected_key,
            )
            self.assertIsInstance(consumer._workgraph_executor, PhysicalExecutor)
            self.assertIsNotNone(consumer._workgraph_plan_store)
            assert consumer._workgraph_plan_store is not None
            self.assertIs(
                consumer._workgraph_plan_store.current.placement_plan,
                activation_resources.placement_plan,
            )

            residual_runtime.prepare_workgraph_step(1)
            with mock.patch.object(
                consumer,
                "_ensure_runtime_resources",
                return_value=None,
            ):
                consumer.consume_bundle(_bundle(source, target), _context())

        legacy_staging.assert_not_called()
        engine.launch_step.assert_not_called()
        self.assertEqual(tuple(label for label, _args in trace), ("selected",))
        self.assertEqual(len(trace[0][1]), 1)
        context = trace[0][1][0]
        self.assertIsInstance(context, PhysicalExecutionContext)
        assert isinstance(context, PhysicalExecutionContext)
        self.assertIs(context.tensors[0], source)
        self.assertIs(context.tensors[1], target)

    def test_clear_active_region_defers_destruction_until_the_prepared_step_retires(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=8))
        _consumer, _phase1, _captured = _esamp_phase1(consumer)
        state = ResidualRuntimeState()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.register_dispatch_consumer(consumer)
            residual_runtime.record_workgraph_backend_resolution(
                resolution=_resolution(),
            )
            residual_runtime.finalize_workgraph_snapshot_after_placement_refresh()
            consumer.prepare_workgraph_step(1)
            store = consumer._workgraph_plan_store
            executor = consumer._workgraph_executor
            self.assertIsNotNone(store)
            self.assertIsNotNone(executor)
            assert store is not None
            active = store.active(1)

            # FULL/disable paths use this same clear API. They must retain the
            # active epoch until its executor records and observes retirement.
            consumer.activate_workgraph_regions(())

            self.assertIs(consumer._workgraph_plan_store, store)
            self.assertIs(consumer._workgraph_executor, executor)
            self.assertIs(store.active(1), active)

            consumer.finish_workgraph_step(1)
            consumer.activate_workgraph_regions(())

        self.assertIsNone(consumer._workgraph_plan_store)
        self.assertIsNone(consumer._workgraph_executor)

    def test_wrapped_prepare_inputs_prepares_physical_regions_before_prepare_dispatch(self) -> None:
        state = ResidualRuntimeState()
        state.event_step_id = 17
        state.launch_consumer_from_hooks = True
        state.fast_weight_phase = "decode"
        order: list[tuple[str, object]] = []
        runner = SimpleNamespace(model=None)

        def _dispatch(**kwargs: object) -> int:
            order.append(("dispatch", kwargs["event_name"]))
            return 0

        fake_core = SimpleNamespace(
            RUNTIME=state,
            MODEL_HOOK_FLAG="_tllm_hook_installed",
            _ORIG_PREPARE_INPUTS=lambda _runner, _scheduler: object(),
            _runner_uses_compilation_or_cudagraph=lambda _runner: False,
        )
        with mock.patch.object(
            port_runtime_hooks.active_targets,
            "runtime_has_active_targets",
            return_value=True,
        ), mock.patch.object(
            residual_runtime,
            "RUNTIME",
            state,
        ), mock.patch.object(
            residual_runtime,
            "_ORIG_PREPARE_INPUTS",
            lambda _runner, _scheduler: object(),
        ), mock.patch.object(
            residual_runtime,
            "prepare_physical_workgraph_step",
            side_effect=lambda step: order.append(("prepare", step)),
        ), mock.patch.object(
            port_runtime_hooks._common_hooks,
            "dispatch_runtime_event",
            side_effect=_dispatch,
        ), mock.patch.object(
            port_runtime_hooks,
            "setup_runtime_hooks_if_active",
        ), mock.patch.object(
            port_runtime_hooks,
            "_clear_decode_localization",
        ), mock.patch.object(
            port_runtime_hooks._sampler_patch,
            "maybe_prepare_sampler_decode_step",
        ), mock.patch.object(
            port_runtime_hooks._fast_weight,
            "update_fast_weight_phase_from_runner",
        ):
            port_runtime_hooks.wrapped_prepare_inputs(
                core=residual_runtime,
                runner=runner,
                scheduler_output=object(),
            )

            self.assertEqual(order[:2], [("prepare", 18), ("dispatch", "prepare_inputs.pre")])
            order.clear()
            port_runtime_hooks.wrapped_prepare_inputs(
                core=fake_core,
                runner=runner,
                scheduler_output=object(),
            )

        self.assertEqual(order, [("dispatch", "prepare_inputs.pre")])

    def test_switching_piecewise_region_to_full_disables_physical_execution_but_keeps_legacy_engine(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(graph_scratch_rows=8),
            engine=engine,
        )
        consumer.specialize_workgraph(
            torch.ones(2, 4),
            torch.zeros(2, 4),
        )
        state = ResidualRuntimeState()
        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.register_dispatch_consumer(consumer)
            residual_runtime.record_workgraph_backend_resolution(
                resolution=_resolution(),
            )
            residual_runtime.finalize_workgraph_snapshot_after_placement_refresh()
            residual_runtime.prepare_workgraph_step(1)

            source = torch.randn(2, 4)
            target = torch.randn(2, 4)
            with mock.patch.object(
                consumer,
                "_ensure_runtime_resources",
                return_value=None,
            ):
                consumer.consume_bundle(
                    _step_bundle(source, target, step_id=1),
                    _context(),
                )

            piecewise_stats = consumer.physical_execution_stats()
            self.assertEqual(piecewise_stats.execute_count, 1)
            self.assertEqual(piecewise_stats.fallback_count, 0)
            engine.launch_step.assert_called_once_with(source, target)

            full_resolution = BackendResolution(
                capabilities=_resolution().capabilities,
                graph_mode=GraphMode.FULL,
            )
            residual_runtime.record_workgraph_backend_resolution(
                resolution=full_resolution,
            )
            residual_runtime.finalize_workgraph_snapshot_after_placement_refresh()

            self.assertIsNone(state.physical_plan_store)
            # FULL clears the runtime-wide physical publication immediately,
            # but an active consumer step still owns its private epoch.  The
            # deactivation is pending until that step records retirement.
            self.assertIsNotNone(consumer._workgraph_plan_store)
            self.assertIsNotNone(consumer._workgraph_executor)
            self.assertTrue(consumer._workgraph_deactivate_pending)
            self.assertEqual(consumer._workgraph_prepared_step_id, 1)
            self.assertIsNotNone(consumer.workgraph_execution_stats())
            with mock.patch.object(
                consumer,
                "_ensure_runtime_resources",
                return_value=None,
            ):
                consumer.consume_bundle(
                    _step_bundle(torch.randn(2, 4), torch.randn(2, 4), step_id=2),
                    _context(),
                )

            consumer.finish_workgraph_step(1)
            self.assertIsNone(consumer._workgraph_plan_store)
            self.assertIsNone(consumer._workgraph_executor)

        self.assertEqual(piecewise_stats.execute_count, 1)
        self.assertEqual(piecewise_stats.fallback_count, 0)
        self.assertEqual(engine.launch_step.call_count, 2)

    def test_region_lowering_selects_only_capture_task_and_identity_is_main_eager(self) -> None:
        consumer, phase1, captured = _esamp_phase1()
        region = _compile_region(self, phase1, captured.artifact_key)
        plan = _region_plan(self, region)

        self.assertEqual(plan.plan_epoch, phase1.epoch)
        self.assertEqual(len(plan.nodes), 1)
        node = plan.nodes[0]
        self.assertEqual(node.node_key.namespace, "capture_task")
        self.assertEqual(node.executor, ExecutorKey("capture", captured.artifact_key.value))
        self.assertEqual(node.stream, StreamClass.SIDE_FORWARD)
        self.assertEqual(node.artifact_key, captured.artifact_key)
        self.assertEqual(node.capture_policy, CapturePolicy.DISABLED)
        self.assertEqual(node.fallback_policy, FallbackPolicy.EAGER)
        self.assertFalse(any(item.executor.namespace == "legacy" for item in plan.nodes))
        self.assertNotEqual(node.node_key, NodeKey.consumer_flow(consumer.consumer_id, 0))

    def test_unknown_and_opaque_artifacts_are_typed_region_rejections(self) -> None:
        _consumer, phase1, captured = _esamp_phase1()
        unknown = ArtifactKey("capture", "unknown", digest="b" * 64)

        with self.assertRaises(WorkGraphError):
            _compile_region(self, phase1, unknown)

        @flow()
        def opaque() -> torch.Tensor:
            print("opaque")
            return torch.ones(2)

        opaque_capture = opaque.capture()
        self.assertEqual(opaque_capture.status, CaptureStatus.OPAQUE_BARRIER)
        # The API must reject a known opaque root with a typed WorkGraph error;
        # it must not manufacture a partial capture task.
        opaque_consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=8))
        opaque_dispatch = DispatchPlan.build([opaque_consumer])
        opaque_phase1 = compile_shadow_parity(
            consumers=(opaque_consumer,),
            dispatch_plan=opaque_dispatch,
            execution_plan=ExecutionPlan(producers=(), port_sources={}),
            backend=shadow_backend_snapshot(),
            captures=(opaque_capture,),
            attachments=(
                CaptureAttachment(
                    node_key=NodeKey.consumer_flow(opaque_consumer.consumer_id, 0),
                    artifact_key=opaque_capture.artifact_key,
                ),
            ),
        )
        with self.assertRaises(WorkGraphError):
            _compile_region(self, opaque_phase1, opaque_capture.artifact_key)

    def test_concrete_activation_uses_base_consumer_polymorphic_region_hook(self) -> None:
        self.assertTrue(
            callable(getattr(BaseConsumer, "activate_workgraph_regions", None)),
            "BaseConsumer must declare the explicit region activation hook",
        )
        consumer = _RecordingESampConsumer(ESampConsumerConfig(graph_scratch_rows=8))
        _consumer, phase1, captured = _esamp_phase1(consumer)
        state = ResidualRuntimeState()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.register_dispatch_consumer(consumer)
            residual_runtime.record_workgraph_backend_resolution(
                resolution=_resolution(),
            )
            residual_runtime.finalize_workgraph_snapshot_after_placement_refresh()

        self.assertEqual(len(consumer.activated_regions), 1)
        region = consumer.activated_regions[0]
        plan = _region_plan(self, region)
        self.assertIsNotNone(state.workgraph_snapshot)
        assert state.workgraph_snapshot is not None
        self.assertEqual(plan.plan_epoch, state.workgraph_snapshot.epoch)
        self.assertEqual(plan.nodes[0].artifact_key, captured.artifact_key)

    def test_physical_handler_performs_training_launch_without_outer_consume(self) -> None:
        """The physical executor must own the real ESamp launch.

        A pure identity capture is only a data handoff description.  Calling
        the activated executor directly must still enter the authoritative
        training engine; otherwise an outer ``consume_bundle`` launch can
        make the physical path look active while the region itself does no
        training.
        """

        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(graph_scratch_rows=8),
            engine=engine,
        )
        _consumer, _phase1, _captured = _esamp_phase1(consumer)
        state = ResidualRuntimeState()
        source = torch.randn(2, 4)
        target = torch.randn(2, 4)

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.register_dispatch_consumer(consumer)
            residual_runtime.record_workgraph_backend_resolution(
                resolution=_resolution(),
            )
            residual_runtime.finalize_workgraph_snapshot_after_placement_refresh()
            consumer.prepare_workgraph_step(1)
            executor = getattr(consumer, "_workgraph_executor", None)
            self.assertIsInstance(executor, PhysicalExecutor)
            assert isinstance(executor, PhysicalExecutor)
            with mock.patch.object(
                consumer,
                "_ensure_runtime_resources",
                return_value=None,
            ):
                executor.execute_step(1, tensors=(source, target))
            consumer.finish_workgraph_step(1)

        engine.launch_step.assert_called_once_with(source, target)

    def test_physical_region_exposes_frozen_typed_training_executor_contract(self) -> None:
        """The capture task must not be mistaken for the stateful trainer."""

        consumer, phase1, captured = _esamp_phase1()
        executors = getattr(consumer, "workgraph_region_executors", None)
        self.assertTrue(
            callable(executors),
            "migrated consumers must expose typed region executor descriptors",
        )
        assert callable(executors)
        executor_descs = executors()
        self.assertEqual(len(executor_descs), 1)
        compiler = getattr(physical_module, "compile_physical_capture_region", None)
        self.assertTrue(callable(compiler))
        assert callable(compiler)
        region = compiler(
            phase1,
            captured.artifact_key,
            executor_desc=executor_descs[0],
        )
        contract = getattr(region, "executor_contract", None)
        self.assertIsInstance(
            contract,
            ExternContract,
            "physical capture regions need an explicit typed executor contract",
        )
        assert isinstance(contract, ExternContract)
        self.assertIs(contract.effect, Effect.WRITE_STATE)
        self.assertIs(contract.state, StateScope.REPLICA)
        self.assertIsInstance(contract.device, Device)
        self.assertIs(contract.capability, Capability.EAGER_ONLY)
        self.assertGreaterEqual(len(contract.inputs), 2)
        with self.assertRaises(FrozenInstanceError):
            contract.effect = Effect.PURE  # type: ignore[misc]

        node = _region_plan(self, region).nodes[0]
        self.assertIsInstance(node.executor, ExecutorKey)
        self.assertEqual(node.executor, executor_descs[0].executor)

    def test_real_esamp_prepare_then_consume_uses_activated_region_and_exposes_physical_stats(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(graph_scratch_rows=8),
            engine=engine,
        )
        _consumer, _phase1, _captured = _esamp_phase1(consumer)
        state = ResidualRuntimeState()
        source = torch.randn(2, 4)
        target = torch.randn(2, 4)

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.register_dispatch_consumer(consumer)
            residual_runtime.record_workgraph_backend_resolution(
                resolution=_resolution(),
            )
            residual_runtime.finalize_workgraph_snapshot_after_placement_refresh()

            stats_before = consumer.workgraph_execution_stats()
            self.assertIsNotNone(stats_before)
            assert stats_before is not None
            residual_runtime.prepare_workgraph_step(1)
            with mock.patch.object(
                consumer,
                "_ensure_runtime_resources",
                return_value=None,
            ):
                consumer.consume_bundle(_bundle(source, target), _context())
            stats_after = consumer.workgraph_execution_stats()

        self.assertIsNotNone(stats_after)
        assert stats_after is not None
        self.assertEqual(stats_after.execute_count, stats_before.execute_count + 1)
        self.assertEqual(stats_after.eager_count, stats_before.eager_count + 1)
        self.assertEqual(stats_after.fallback_count, 0)
        engine.launch_step.assert_called_once_with(source, target)

    def test_prepared_region_executes_consume_bundle_with_original_tensors_without_fallback(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(graph_scratch_rows=8),
            engine=engine,
        )
        _consumer, phase1, captured = _esamp_phase1(consumer)
        region = _compile_region(self, phase1, captured.artifact_key)
        plan = _region_plan(self, region)
        resources = PlanResources(
            plan=plan,
            streams=(_runtime_stream(),),
        )
        executor = PhysicalExecutor(
            snapshots=PlanSnapshotStore(resources),
            handlers={
                plan.nodes[0].executor: lambda physical_ctx: consumer.consume_bundle(
                    _bundle(physical_ctx.tensors[0], physical_ctx.tensors[1]),
                    _context(),
                ),
            },
        )
        source = torch.randn(2, 4)
        target = torch.randn(2, 4)
        with mock.patch.object(consumer, "_ensure_runtime_resources", return_value=None):
            executor.prepare_step(1)
            result = executor.execute_step(1, tensors=(source, target))

        self.assertEqual(result.eager_nodes, (str(plan.nodes[0].node_key),))
        self.assertEqual(result.replayed_nodes, ())
        self.assertEqual(executor.stats.execute_count, 1)
        self.assertEqual(executor.stats.eager_count, 1)
        self.assertEqual(executor.stats.fallback_count, 0)
        engine.launch_step.assert_called_once_with(source, target)
        executor.finish_step(1)

    def test_unprepared_consume_bundle_keeps_legacy_direct_path_and_never_captures_hot(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(graph_scratch_rows=8),
            engine=engine,
        )
        source = torch.randn(2, 4)
        target = torch.randn(2, 4)
        bundle = _bundle(source, target)
        with mock.patch.object(consumer, "_ensure_runtime_resources", return_value=None), mock.patch(
            "tllm.workgraph.capture.capture_flow",
            side_effect=AssertionError("hot consume must not capture"),
        ):
            consumer.consume_bundle(bundle, _context())

        engine.launch_step.assert_called_once_with(source, target)

        # A physical executor must not silently execute before its cold
        # prepare gate has acquired a plan snapshot.
        _consumer, phase1, captured = _esamp_phase1(consumer)
        region = _compile_region(self, phase1, captured.artifact_key)
        plan = _region_plan(self, region)
        executor = PhysicalExecutor(
            snapshots=PlanSnapshotStore(
                PlanResources(
                    plan=plan,
                    streams=(_runtime_stream(),),
                )
            ),
            handlers={plan.nodes[0].executor: lambda _ctx: None},
        )
        with self.assertRaises(PhysicalExecutionError):
            executor.execute_step(1, tensors=(source, target))

    def test_disabled_physical_region_uses_one_legacy_training_launch(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(
                graph_scratch_rows=8,
                enable_workgraph_execution=False,
            ),
            engine=engine,
        )
        source = torch.randn(2, 4)
        target = torch.randn(2, 4)

        with mock.patch.object(consumer, "_ensure_runtime_resources", return_value=None):
            consumer.consume_bundle(_bundle(source, target), _context())

        engine.launch_step.assert_called_once_with(source, target)


if __name__ == "__main__":
    unittest.main()

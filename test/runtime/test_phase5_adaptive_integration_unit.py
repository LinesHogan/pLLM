"""Production-path CPU contracts for Phase 5 adaptive queue admission."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tllm.consumers.base import BaseConsumer
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.base import ConsumerFlow, TimingContract
from tllm.runtime import residual_runtime
from tllm.runtime.dispatch_plan import FlowDispatchTarget
from tllm.runtime.ports.residual_bundle_dispatch import (
    _enqueue_flow_target,
    drain_deferred_port_bundles,
    synchronize_deferred_port_bundles,
)
from tllm.runtime.scheduling.deferred import DeferredWork
from tllm.runtime.scheduling.phase3_transfer import (
    PinnedMemoryPool,
    PlacementPlanDispatcher,
    PlacementRuntimeBinding,
)
from tllm.runtime.scheduling.phase5_adaptive import (
    AdaptiveFlowBinding,
    AdaptiveRuntimeBinding,
    AdaptiveTransferBinding,
    Phase5AdaptiveScheduler,
)
from tllm.runtime.scheduling.plan_store import PlanResources, PlanSnapshotStore
from tllm.runtime.state import ResidualRuntimeState
from tllm.workgraph import phase3
from tllm.workgraph.ir import ExecutorKey, NodeKey
from tllm.workgraph.phase3 import DeviceTopology, PlacementPlan
from tllm.workgraph.phase5 import (
    AdmissionMode,
    CandidateActionDesc,
    CandidateActionKey,
    CandidateActionKind,
    CandidateMoveSetDesc,
    MoveSetKey,
    Phase5ControlPlan,
    ShapeBucketDesc,
    ShapeBucketKey,
    WorkItemDesc,
    WorkItemKey,
    WorkloadBucketDesc,
    WorkloadBucketKey,
)
from tllm.workgraph.physical import (
    PhysicalFlowPlan,
    PhysicalNodePlan,
    StreamClass,
)


class _UnusedFence:
    def query(self) -> bool:
        return True


class _UnusedCopyAdapter:
    def allocate_pinned(self, shape: tuple[int, ...], dtype: str) -> object:
        del dtype
        return torch.empty(shape)

    def copy(
        self,
        kind: str,
        source: object,
        destination: object,
        *,
        staging: object | None = None,
    ) -> _UnusedFence:
        del kind, source, destination, staging
        raise AssertionError("empty placement plan must not submit a copy")


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...], dtype: str, device: str) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device


class _TrackingFence:
    def __init__(self) -> None:
        self.complete = False

    def query(self) -> bool:
        return self.complete


class _TrackingCopyAdapter:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fences: list[_TrackingFence] = []

    def allocate_pinned(self, shape: tuple[int, ...], dtype: str) -> _FakeTensor:
        return _FakeTensor(shape, dtype, "cpu:pinned")

    def copy(
        self,
        kind: str,
        source: object,
        destination: object,
        *,
        staging: object | None = None,
    ) -> _TrackingFence:
        del source, destination, staging
        self.calls.append(kind)
        fence = _TrackingFence()
        self.fences.append(fence)
        return fence


class _Implementation:
    def execute(self, *args: object) -> None:
        del args


def _runtime_resources(
    *,
    item: WorkItemDesc,
    mode: AdmissionMode,
    flow_id: str = "phase5.flow",
    include_shed: bool = False,
) -> tuple[PlanResources, CandidateActionKey]:
    placement = PlacementPlan(
        plan_epoch=1,
        selections=(),
        schedule=(),
        topology=DeviceTopology(frozenset()),
    )
    adapter = _UnusedCopyAdapter()
    dispatcher = PlacementPlanDispatcher(
        plan=placement,
        pool=PinnedMemoryPool(capacity=1, adapter=adapter),
        adapter=adapter,
    )
    shape_key = ShapeBucketKey("phase5", "shape")
    workload_key = WorkloadBucketKey("phase5", "decode")
    fallback_key = CandidateActionKey("phase5", "fallback")
    execute_key = CandidateActionKey("phase5", "execute")
    fallback = CandidateActionDesc(
        fallback_key,
        CandidateActionKind.AUTHORITATIVE_FALLBACK,
        unknown_bucket=True,
    )
    execute = CandidateActionDesc(
        execute_key,
        CandidateActionKind.EXECUTE,
        workload_bucket=workload_key,
        shape_bucket=shape_key,
        fallback_key=fallback_key,
        deadline=4,
        duration=1,
    )
    shed = CandidateActionDesc(
        CandidateActionKey("phase5", "shed"),
        CandidateActionKind.SHED,
        workload_bucket=workload_key,
        shape_bucket=shape_key,
        duration=0,
    )
    actions = (fallback, execute, shed) if include_shed else (fallback, execute)
    plan = Phase5ControlPlan(
        plan_epoch=1,
        placement_plan=placement,
        scheduled_moves=(),
        move_sets=(),
        workload_buckets=(WorkloadBucketDesc(workload_key, (shape_key,)),),
        shape_buckets=(ShapeBucketDesc(shape_key, shape=(2,)),),
        budgets=(),
        actions=actions,
        unknown_bucket_fallback=fallback_key,
        buffer_capacity=1,
        mode=mode,
    )
    scheduler = Phase5AdaptiveScheduler(plan)
    return (
        PlanResources(
            plan=PhysicalFlowPlan(plan_epoch=1, nodes=()),
            placement_plan=placement,
            placement_dispatcher=dispatcher,
            adaptive_plan=plan,
            adaptive_scheduler=scheduler,
            adaptive_flow_bindings=(AdaptiveFlowBinding(flow_id, item),),
        ),
        execute_key,
    )


def _queued(
    flow_id: str,
    trace: list[str],
    *,
    entries: dict[str, object] | None = None,
    deadline_step: int = 8,
    plan_resources: PlanResources | None = None,
) -> DeferredWork:
    return DeferredWork(
        flow_id=flow_id,
        bundle=PortBundle(
            key=BundleKey(7, "decode", "request", 0),
            entries={} if entries is None else entries,
            version=7,
        ),
        context=RuntimeContext(
            runner=None,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="phase5",
        ),
        run=lambda _bundle, _context: trace.append("run"),
        enqueued_step=7,
        eligible_step=7,
        eligible_hook="sampler.pre_draw",
        deadline_step=deadline_step,
        deadline_hook="prepare_inputs.pre",
        plan_resources=plan_resources,
    )


def _transfer_resources(
    *,
    epoch: int = 1,
    adapter: _TrackingCopyAdapter | None = None,
    include_binding: bool = True,
    fallback_move_set: bool = False,
) -> tuple[PlanResources, _TrackingCopyAdapter]:
    """Build one cold adaptive resource epoch with a two-hop transfer route."""

    source_node = NodeKey.producer("phase5.transfer.source")
    target_node = NodeKey.consumer_flow("phase5.transfer.target", 0)
    source_logical = ExecutorKey("logical", "phase5.transfer.source")
    target_logical = ExecutorKey("logical", "phase5.transfer.target")
    source_impl = ExecutorKey("implementation", "phase5.transfer.source.cuda0")
    target_impl = ExecutorKey("implementation", "phase5.transfer.target.cuda1")
    physical = PhysicalFlowPlan(
        plan_epoch=epoch,
        nodes=(
            PhysicalNodePlan(source_node, source_logical, StreamClass.DEFAULT),
            PhysicalNodePlan(target_node, target_logical, StreamClass.SIDE_TRAIN),
        ),
    )
    registry = phase3.ImplementationRegistry()
    registry.register(
        phase3.ImplementationSpec(
            source_impl,
            phase3.Placement("cuda", 0),
            _Implementation(),
            logical_executor=source_logical,
        )
    )
    registry.register(
        phase3.ImplementationSpec(
            target_impl,
            phase3.Placement("cuda", 1),
            _Implementation(),
            logical_executor=target_logical,
        )
    )
    value = phase3.ValueKey("phase5-transfer", 1)
    if adapter is None:
        adapter = _TrackingCopyAdapter()
    placement, dispatcher, implementations = PlacementRuntimeBinding(
        registry=registry,
        topology=phase3.DeviceTopology(frozenset(), (0, 1)),
        value_flows=(phase3.ValueFlowDesc(value, source_node, target_node, 16),),
        adapter=adapter,
    ).compile(physical)
    move_set = CandidateMoveSetDesc(
        MoveSetKey("phase5", "transfer"),
        placement.moves,
    )
    route_deadline = max(move.end_at for move in placement.moves) + 1
    shape_key = ShapeBucketKey("phase5", "transfer-shape")
    workload_key = WorkloadBucketKey("phase5", "transfer-work")
    fallback_key = CandidateActionKey("phase5", "transfer-fallback")
    execute_key = CandidateActionKey("phase5", "transfer-execute")
    fallback = CandidateActionDesc(
        fallback_key,
        CandidateActionKind.AUTHORITATIVE_FALLBACK,
        move_set=move_set.key if fallback_move_set else None,
        deadline=route_deadline if fallback_move_set else 1,
        unknown_bucket=True,
    )
    execute = CandidateActionDesc(
        execute_key,
        CandidateActionKind.EXECUTE,
        workload_bucket=workload_key,
        shape_bucket=shape_key,
        move_set=move_set.key,
        fallback_key=fallback_key,
        deadline=route_deadline,
    )
    plan = Phase5ControlPlan(
        plan_epoch=epoch,
        placement_plan=placement,
        scheduled_moves=placement.moves,
        move_sets=(move_set,),
        workload_buckets=(WorkloadBucketDesc(workload_key, (shape_key,)),),
        shape_buckets=(ShapeBucketDesc(shape_key, shape=(4,)),),
        budgets=(),
        actions=(fallback, execute),
        unknown_bucket_fallback=fallback_key,
        buffer_capacity=1,
        mode=AdmissionMode.ADAPTIVE,
    )
    item = WorkItemDesc(
        WorkItemKey("phase5", "transfer-item"),
        (99,) if fallback_move_set else (4,),
        workload_key,
    )
    flow_binding = AdaptiveFlowBinding(
        "phase5.transfer.flow",
        item,
        ()
        if not include_binding
        else (
            AdaptiveTransferBinding(
                move_set.key,
                source_node,
                target_node,
                value,
                "source",
                "destination",
            ),
        ),
    )
    resources = PlanResources(
        plan=physical,
        placement_plan=placement,
        placement_dispatcher=dispatcher,
        placement_implementations=implementations,
        adaptive_plan=plan,
        adaptive_scheduler=Phase5AdaptiveScheduler(plan),
        adaptive_flow_bindings=(flow_binding,),
    )
    return resources, adapter


def _enqueue_production_transfer(
    state: ResidualRuntimeState,
    trace: list[str],
) -> None:
    flow = ConsumerFlow(
        reads=(),
        writes=(),
        window="background",
        bundle_key=("engine_step_id", "phase"),
        timing=TimingContract(
            mobility="deferrable",
            deadline="next_prepare",
            max_defer_steps=1,
        ),
    )
    consumer = type(
        "QueuedTransferConsumer",
        (BaseConsumer,),
        {
            "consumer_id": property(lambda self: "phase5.transfer.consumer"),
            "consume_bundle": lambda self, bundle, ctx: trace.append("run"),
            "flows": lambda self: (flow,),
        },
    )()
    target = FlowDispatchTarget(
        consumer=consumer,
        flow=flow,
        flow_id="phase5.transfer.flow",
        placement_hook="sampler.pre_draw",
    )
    queued = _queued(
        target.flow_id,
        trace,
        entries={
            "source": _FakeTensor((4,), "float32", "cuda:0"),
            "destination": _FakeTensor((4,), "float32", "cuda:1"),
        },
    )
    _enqueue_flow_target(
        queue=state.deferred_work_queue,
        runtime=state,
        target=target,
        bundle=queued.bundle,
        ctx=queued.context,
        step_id=7,
    )


class Phase5AdaptiveIntegrationUnitTest(unittest.TestCase):
    def _state(self, resources: PlanResources) -> ResidualRuntimeState:
        state = ResidualRuntimeState()
        state.physical_plan_store = PlanSnapshotStore(resources)
        state.physical_plan_store.prepare_step(7)
        state.physical_prepared_step_id = 7
        return state

    def test_cold_configuration_atomically_publishes_adaptive_resources(self) -> None:
        item = WorkItemDesc(
            WorkItemKey("phase5", "configured"),
            (2,),
            WorkloadBucketKey("phase5", "decode"),
        )
        template, _execute_key = _runtime_resources(
            item=item,
            mode=AdmissionMode.ADAPTIVE,
        )
        plan = template.adaptive_plan
        placement = template.placement_plan
        assert plan is not None
        assert placement is not None
        binding = AdaptiveRuntimeBinding(
            move_sets=plan.move_sets,
            workload_buckets=plan.workload_buckets,
            shape_buckets=plan.shape_buckets,
            budgets=plan.budgets,
            actions=plan.actions,
            unknown_bucket_fallback=plan.unknown_bucket_fallback,
            flow_bindings=(AdaptiveFlowBinding("phase5.flow", item),),
            profiles=plan.profiles,
            buffer_capacity=plan.buffer_capacity,
            mode=plan.mode,
            densification_limit=plan.densification_limit,
        )
        placement_binding = mock.Mock()
        placement_binding.compile.return_value = (
            placement,
            template.placement_dispatcher,
            (),
        )
        physical = PhysicalFlowPlan(plan_epoch=placement.plan_epoch, nodes=())
        state = ResidualRuntimeState()
        state.workgraph_snapshot = SimpleNamespace(
            backend=SimpleNamespace(graph_mode="eager")
        )
        state.placement_runtime_binding = placement_binding

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch(
                "tllm.workgraph.physical.compile_physical_flow_plan",
                return_value=physical,
            ),
            mock.patch.object(residual_runtime, "_activate_physical_capture_regions"),
        ):
            residual_runtime.configure_workgraph_adaptive(binding)
            residual_runtime.prepare_workgraph_step(7)

        store = state.physical_plan_store
        self.assertIsNotNone(store)
        assert store is not None
        resources = store.active(7)
        self.assertIs(resources.plan, physical)
        self.assertIs(resources.placement_plan, placement)
        self.assertIsNotNone(resources.adaptive_plan)
        self.assertIsNotNone(resources.adaptive_scheduler)
        assert resources.adaptive_plan is not None
        self.assertIs(resources.adaptive_plan.placement_plan, placement)
        self.assertEqual(resources.adaptive_plan.plan_epoch, physical.plan_epoch)
        self.assertEqual(resources.adaptive_flow_bindings, binding.flow_bindings)

    def test_static_prepared_epoch_ignores_ewma_and_runs_bound_flow(self) -> None:
        item = WorkItemDesc(
            WorkItemKey("phase5", "optional"),
            (2,),
            WorkloadBucketKey("phase5", "decode"),
        )
        resources, execute_key = _runtime_resources(
            item=item,
            mode=AdmissionMode.STATIC,
        )
        assert resources.adaptive_scheduler is not None
        resources.adaptive_scheduler.observe(execute_key, 100)
        state = self._state(resources)
        trace: list[str] = []
        state.deferred_work_queue.enqueue(_queued("phase5.flow", trace))

        drained = drain_deferred_port_bundles(
            runtime=state,
            step_id=7,
            hook="sampler.pre_draw",
        )

        self.assertEqual(drained, 1)
        self.assertEqual(trace, ["run"])
        self.assertEqual(state.placement_stats.adaptive_run_count, 1)

    def test_adaptive_optional_work_defers_then_sheds_at_deadline(self) -> None:
        item = WorkItemDesc(
            WorkItemKey("phase5", "optional"),
            (2,),
            WorkloadBucketKey("phase5", "decode"),
        )
        resources, execute_key = _runtime_resources(
            item=item,
            mode=AdmissionMode.ADAPTIVE,
            include_shed=True,
        )
        assert resources.adaptive_scheduler is not None
        resources.adaptive_scheduler.observe(execute_key, 100)
        state = self._state(resources)
        trace: list[str] = []
        state.deferred_work_queue.enqueue(_queued("phase5.flow", trace))

        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=7,
                hook="sampler.pre_draw",
            ),
            0,
        )
        self.assertEqual(state.deferred_work_queue.depth, 1)
        state.physical_plan_store.finish_step(7)
        state.physical_plan_store.prepare_step(8)
        state.physical_prepared_step_id = 8
        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=8,
                hook="prepare_inputs.pre",
                expired_only=True,
            ),
            0,
        )

        self.assertFalse(state.deferred_work_queue)
        self.assertEqual(trace, [])
        self.assertEqual(state.placement_stats.adaptive_defer_count, 1)
        self.assertEqual(state.placement_stats.adaptive_shed_count, 1)

    def test_optional_work_falls_back_when_shedding_was_not_approved(self) -> None:
        item = WorkItemDesc(
            WorkItemKey("phase5", "optional-fallback"),
            (2,),
            WorkloadBucketKey("phase5", "decode"),
        )
        resources, execute_key = _runtime_resources(
            item=item,
            mode=AdmissionMode.ADAPTIVE,
        )
        assert resources.adaptive_scheduler is not None
        resources.adaptive_scheduler.observe(execute_key, 100)
        state = self._state(resources)
        trace: list[str] = []
        state.deferred_work_queue.enqueue(_queued("phase5.flow", trace))

        drained = drain_deferred_port_bundles(
            runtime=state,
            step_id=7,
            hook="sampler.pre_draw",
            expired_only=True,
        )

        self.assertEqual(drained, 0)
        self.assertEqual(trace, [])
        self.assertEqual(state.deferred_work_queue.depth, 1)

        state.physical_plan_store.finish_step(7)
        state.physical_plan_store.prepare_step(8)
        state.physical_prepared_step_id = 8
        drained = drain_deferred_port_bundles(
            runtime=state,
            step_id=8,
            hook="prepare_inputs.pre",
            expired_only=True,
        )
        self.assertEqual(drained, 1)
        self.assertEqual(trace, ["run"])
        self.assertEqual(state.placement_stats.adaptive_fallback_count, 1)
        self.assertEqual(state.placement_stats.adaptive_shed_count, 0)

    def test_required_work_uses_authoritative_fallback_instead_of_drop(self) -> None:
        item = WorkItemDesc(
            WorkItemKey("phase5", "required"),
            (2,),
            WorkloadBucketKey("phase5", "decode"),
            required=True,
        )
        resources, execute_key = _runtime_resources(
            item=item,
            mode=AdmissionMode.ADAPTIVE,
        )
        assert resources.adaptive_scheduler is not None
        resources.adaptive_scheduler.observe(execute_key, 100)
        state = self._state(resources)
        trace: list[str] = []
        state.deferred_work_queue.enqueue(_queued("phase5.flow", trace))

        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=7,
                hook="sampler.pre_draw",
            ),
            1,
        )
        self.assertEqual(trace, ["run"])
        self.assertEqual(state.placement_stats.adaptive_fallback_count, 1)
        self.assertEqual(state.placement_stats.adaptive_shed_count, 0)

    def test_cold_plan_resources_reject_selectable_move_set_without_transfer_binding(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly cover selectable non-empty MoveSets"):
            _transfer_resources(include_binding=False)

    def test_pending_transfer_remains_owned_by_old_epoch_until_callback(self) -> None:
        old_resources, old_adapter = _transfer_resources()
        new_resources, new_adapter = _transfer_resources(epoch=2)
        state = self._state(old_resources)
        trace: list[str] = []
        state.deferred_work_queue.enqueue(
            _queued(
                "phase5.transfer.flow",
                trace,
                entries={
                    "source": _FakeTensor((4,), "float32", "cuda:0"),
                    "destination": _FakeTensor((4,), "float32", "cuda:1"),
                },
                deadline_step=12,
            )
        )

        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=7,
                hook="sampler.pre_draw",
            ),
            0,
        )
        self.assertEqual(old_adapter.calls, ["d2h"])
        self.assertEqual(trace, [])

        store = state.physical_plan_store
        assert store is not None
        store.finish_step(7)
        store.stage(new_resources)
        store.prepare_step(8)
        state.physical_prepared_step_id = 8
        binding = old_resources.adaptive_flow_binding("phase5.transfer.flow")
        assert binding is not None
        self.assertIs(
            store.adaptive_transfer_owner(
                flow_id=binding.flow_id,
                bundle_key=BundleKey(7, "decode", "request", 0),
                bundle_version=7,
            ),
            old_resources,
        )

        old_adapter.fences[0].complete = True
        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=8,
                hook="sampler.pre_draw",
            ),
            0,
        )
        self.assertEqual(old_adapter.calls, ["d2h", "h2d"])
        self.assertEqual(new_adapter.calls, [])
        self.assertEqual(trace, [])

        old_adapter.fences[1].complete = True
        self.assertEqual(store.collect_retired(), 0)
        self.assertIs(
            store.adaptive_transfer_owner(
                flow_id=binding.flow_id,
                bundle_key=BundleKey(7, "decode", "request", 0),
                bundle_version=7,
            ),
            old_resources,
        )
        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=8,
                hook="sampler.pre_draw",
            ),
            1,
        )
        self.assertEqual(trace, ["run"])
        self.assertEqual(store.collect_retired(), 1)

    def test_legacy_enqueue_is_not_retroactively_adapted_by_new_epoch(self) -> None:
        legacy_resources = PlanResources(
            plan=PhysicalFlowPlan(plan_epoch=1, nodes=()),
        )
        adaptive_resources, adapter = _transfer_resources(epoch=2)
        state = self._state(legacy_resources)
        trace: list[str] = []
        _enqueue_production_transfer(state, trace)

        store = state.physical_plan_store
        assert store is not None
        store.finish_step(7)
        store.stage(adaptive_resources)
        store.prepare_step(8)
        state.physical_prepared_step_id = 8

        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=8,
                hook="sampler.pre_draw",
            ),
            1,
        )
        self.assertEqual(adapter.calls, [])
        self.assertEqual(trace, ["run"])

    def test_queued_transfer_retains_enqueue_epoch_before_first_drain(self) -> None:
        old_resources, old_adapter = _transfer_resources()
        new_resources, new_adapter = _transfer_resources(epoch=2)
        state = self._state(old_resources)
        trace: list[str] = []
        _enqueue_production_transfer(state, trace)

        store = state.physical_plan_store
        assert store is not None
        store.finish_step(7)
        store.stage(new_resources)
        store.prepare_step(8)
        state.physical_prepared_step_id = 8
        prematurely_collected = store.collect_retired()

        def complete_latest_fence(_seconds: float) -> None:
            adapter = old_adapter if old_adapter.fences else new_adapter
            self.assertTrue(adapter.fences)
            adapter.fences[-1].complete = True

        with mock.patch("time.sleep", side_effect=complete_latest_fence):
            synchronize_deferred_port_bundles(runtime=state)
        collected_after_callback = store.collect_retired()

        self.assertEqual(
            (
                prematurely_collected,
                old_adapter.calls,
                new_adapter.calls,
                trace,
                collected_after_callback,
            ),
            (0, ["d2h", "h2d"], [], ["run"], 1),
        )

    def test_owned_old_epoch_uses_typed_drain_when_current_is_non_adaptive(self) -> None:
        old_resources, old_adapter = _transfer_resources()
        state = self._state(old_resources)
        trace: list[str] = []
        _enqueue_production_transfer(state, trace)

        store = state.physical_plan_store
        assert store is not None
        store.finish_step(7)
        store.stage(PlanResources(plan=PhysicalFlowPlan(plan_epoch=2, nodes=())))
        store.prepare_step(8)
        state.physical_prepared_step_id = 8
        self.assertEqual(store.collect_retired(), 0)

        with self.assertRaisesRegex(
            RuntimeError,
            "adaptive Phase 3 transfer missed the bundle deadline",
        ):
            drain_deferred_port_bundles(
                runtime=state,
                step_id=8,
                hook="sampler.pre_draw",
            )
        self.assertEqual(old_adapter.calls, ["d2h"])
        self.assertEqual(trace, [])
        self.assertEqual(state.deferred_work_queue.depth, 1)
        state.deferred_work_queue.clear()

    def test_clear_abandons_pending_transfer_handoff_after_fence_completion(self) -> None:
        old_resources, old_adapter = _transfer_resources()
        new_resources, _new_adapter = _transfer_resources(epoch=2)
        state = self._state(old_resources)
        trace: list[str] = []
        old_resources.retain_deferred_handoff()
        state.deferred_work_queue.enqueue(
            _queued(
                "phase5.transfer.flow",
                trace,
                entries={
                    "source": _FakeTensor((4,), "float32", "cuda:0"),
                    "destination": _FakeTensor((4,), "float32", "cuda:1"),
                },
                plan_resources=old_resources,
            )
        )
        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=7,
                hook="sampler.pre_draw",
            ),
            0,
        )
        self.assertEqual(old_adapter.calls, ["d2h"])

        state.deferred_work_queue.clear()
        store = state.physical_plan_store
        assert store is not None
        store.finish_step(7)
        store.stage(new_resources)
        store.prepare_step(8)
        state.physical_prepared_step_id = 8
        self.assertEqual(store.collect_retired(), 0)

        old_adapter.fences[0].complete = True
        self.assertEqual(store.collect_retired(), 0)
        self.assertEqual(old_adapter.calls, ["d2h", "h2d"])
        old_adapter.fences[1].complete = True
        self.assertEqual(store.collect_retired(), 1)
        self.assertEqual(trace, [])

    def test_authoritative_fallback_completes_its_bound_move_set_before_callback(self) -> None:
        resources, adapter = _transfer_resources(fallback_move_set=True)
        state = self._state(resources)
        trace: list[str] = []
        state.deferred_work_queue.enqueue(
            _queued(
                "phase5.transfer.flow",
                trace,
                entries={
                    "source": _FakeTensor((4,), "float32", "cuda:0"),
                    "destination": _FakeTensor((4,), "float32", "cuda:1"),
                },
            )
        )

        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=7,
                hook="sampler.pre_draw",
            ),
            0,
        )
        self.assertEqual(adapter.calls, ["d2h"])
        self.assertEqual(trace, [])

        adapter.fences[0].complete = True
        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=7,
                hook="sampler.pre_draw",
            ),
            0,
        )
        self.assertEqual(adapter.calls, ["d2h", "h2d"])
        self.assertEqual(trace, [])

        adapter.fences[1].complete = True
        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=7,
                hook="sampler.pre_draw",
            ),
            1,
        )
        self.assertEqual(trace, ["run"])
        self.assertEqual(state.placement_stats.adaptive_fallback_count, 1)
        self.assertEqual(state.placement_stats.adaptive_run_count, 0)

    def test_synchronize_esamp_drains_adaptive_transfer_before_callback(self) -> None:
        resources, adapter = _transfer_resources()
        state = self._state(resources)
        trace: list[str] = []
        state.deferred_work_queue.enqueue(
            _queued(
                "phase5.transfer.flow",
                trace,
                entries={
                    "source": _FakeTensor((4,), "float32", "cuda:0"),
                    "destination": _FakeTensor((4,), "float32", "cuda:1"),
                },
                deadline_step=12,
            )
        )
        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=7,
                hook="sampler.pre_draw",
            ),
            0,
        )
        self.assertEqual(adapter.calls, ["d2h"])
        self.assertEqual(trace, [])

        def complete_latest_fence(_seconds: float) -> None:
            self.assertTrue(adapter.fences)
            adapter.fences[-1].complete = True

        queue_type = type(state.deferred_work_queue)
        with (
            mock.patch.object(
                queue_type,
                "flush_all",
                wraps=state.deferred_work_queue.flush_all,
            ) as flush_all,
            mock.patch("time.sleep", side_effect=complete_latest_fence) as sleep,
            mock.patch.object(residual_runtime, "RUNTIME", state),
        ):
            residual_runtime.synchronize_esamp()

        flush_all.assert_not_called()
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(adapter.calls, ["d2h", "h2d"])
        self.assertEqual(trace, ["run"])

    def test_selected_move_set_runs_exact_phase3_route_before_callback(self) -> None:
        resources, adapter = _transfer_resources()
        placement = resources.placement_plan
        assert placement is not None
        self.assertEqual(len(placement.moves), 2)
        state = self._state(resources)
        trace: list[str] = []
        source = _FakeTensor((4,), "float32", "cuda:0")
        destination = _FakeTensor((4,), "float32", "cuda:1")
        state.deferred_work_queue.enqueue(
            _queued(
                "phase5.transfer.flow",
                trace,
                entries={"source": source, "destination": destination},
            )
        )

        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=7,
                hook="sampler.pre_draw",
            ),
            0,
        )
        self.assertEqual(adapter.calls, ["d2h"])
        self.assertEqual(trace, [])
        adapter.fences[0].complete = True
        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=7,
                hook="sampler.pre_draw",
            ),
            0,
        )
        self.assertEqual(adapter.calls, ["d2h", "h2d"])
        self.assertEqual(trace, [])
        adapter.fences[1].complete = True
        self.assertEqual(
            drain_deferred_port_bundles(
                runtime=state,
                step_id=7,
                hook="sampler.pre_draw",
            ),
            1,
        )
        self.assertEqual(trace, ["run"])
        self.assertEqual(state.placement_stats.adaptive_defer_count, 2)
        self.assertEqual(state.placement_stats.adaptive_run_count, 1)


if __name__ == "__main__":
    unittest.main()

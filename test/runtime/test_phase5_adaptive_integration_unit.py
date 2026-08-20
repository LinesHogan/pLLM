"""Production-path CPU contracts for Phase 5 adaptive queue admission."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.runtime import residual_runtime
from tllm.runtime.ports.residual_bundle_dispatch import drain_deferred_port_bundles
from tllm.runtime.scheduling.deferred import DeferredWork
from tllm.runtime.scheduling.phase3_transfer import (
    PinnedMemoryPool,
    PlacementPlanDispatcher,
)
from tllm.runtime.scheduling.phase5_adaptive import (
    AdaptiveFlowBinding,
    AdaptiveRuntimeBinding,
    Phase5AdaptiveScheduler,
)
from tllm.runtime.scheduling.plan_store import PlanResources, PlanSnapshotStore
from tllm.runtime.state import ResidualRuntimeState
from tllm.workgraph.phase3 import DeviceTopology, PlacementPlan
from tllm.workgraph.phase5 import (
    AdmissionMode,
    CandidateActionDesc,
    CandidateActionKey,
    CandidateActionKind,
    Phase5ControlPlan,
    ShapeBucketDesc,
    ShapeBucketKey,
    WorkItemDesc,
    WorkItemKey,
    WorkloadBucketDesc,
    WorkloadBucketKey,
)
from tllm.workgraph.physical import PhysicalFlowPlan


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


def _runtime_resources(
    *,
    item: WorkItemDesc,
    mode: AdmissionMode,
    flow_id: str = "phase5.flow",
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
    plan = Phase5ControlPlan(
        plan_epoch=1,
        placement_plan=placement,
        scheduled_moves=(),
        move_sets=(),
        workload_buckets=(WorkloadBucketDesc(workload_key, (shape_key,)),),
        shape_buckets=(ShapeBucketDesc(shape_key, shape=(2,)),),
        budgets=(),
        actions=(fallback, execute),
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


def _queued(flow_id: str, trace: list[str]) -> DeferredWork:
    return DeferredWork(
        flow_id=flow_id,
        bundle=PortBundle(
            key=BundleKey(7, "decode", "request", 0),
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
        deadline_step=8,
        deadline_hook="prepare_inputs.pre",
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


if __name__ == "__main__":
    unittest.main()

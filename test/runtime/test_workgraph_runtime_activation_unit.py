"""Failing contracts for WorkGraph runtime activation.

These tests deliberately exercise the cold-path handoff only.  The runtime
must publish a physical ``PlanResources`` snapshot without changing the
authoritative DispatchPlan/ExecutionPlan or making the physical snapshot
visible before the next prepare safe point.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety, SamplerStage
from tllm.consumers.base import BaseConsumer
from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.runtime import residual_runtime
from tllm.runtime.scheduling.plan_store import PlanResources, PlanSnapshotStore
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.graph_policy import GraphMode
from tllm.workgraph.anchors import BackendResolution
from tllm.workgraph.physical import (
    CapturePolicy,
    FallbackPolicy,
    PhysicalFlowPlan,
)
from tllm.workgraph.ir import NodeKey


def _resolution() -> BackendResolution:
    return BackendResolution(
        capabilities=BackendCapabilities(
            backend_id="vllm-phase2-activation",
            runner_family="mrv2",
            sampler_stages=frozenset({SamplerStage.BEFORE_DRAW}),
            native_min_p=True,
            tuple_sampler_return=True,
            split_execute_and_sample=True,
            graph_safety=GraphSafety.CAPTURE_SAFE,
        ),
        graph_mode=GraphMode.PIECEWISE,
    )


def _specialize(consumer: ESampConsumer, rows: int) -> None:
    consumer.specialize_workgraph(
        torch.ones(rows, 4),
        torch.zeros(rows, 4),
    )


def _activate_concrete(
    state: ResidualRuntimeState,
    consumer: ESampConsumer,
) -> tuple[object, object]:
    """Register and complete the existing placement refresh cold path."""

    with mock.patch.object(residual_runtime, "RUNTIME", state):
        residual_runtime.register_dispatch_consumer(consumer)
        dispatch_plan = state.dispatch_plan
        execution_plan = state.execution_plan
        assert dispatch_plan is not None
        assert execution_plan is not None
        residual_runtime.record_workgraph_backend_resolution(
            resolution=_resolution(),
        )
        # ESamp declares a graph-sensitive fusion flow.  In production the
        # dispatch-side placement refresh precedes this cold finalization.
        residual_runtime.finalize_workgraph_snapshot_after_placement_refresh()
        return dispatch_plan, execution_plan


class WorkGraphRuntimeActivationUnitTest(unittest.TestCase):
    def test_prepare_drains_consumer_specializations_before_acquiring_new_snapshot(self) -> None:
        order: list[tuple[str, int]] = []

        class _DrainProbeConsumer(BaseConsumer[object]):
            config = object()

            @property
            def consumer_id(self) -> str:
                return "phase2.drain-probe"

            def drain_workgraph_specializations(self) -> int:
                order.append(("drain", 7))
                return 1

            def prepare_workgraph_step(self, step_id: int) -> None:
                order.append(("consumer_prepare", int(step_id)))

        store = mock.Mock()
        store.prepare_step.side_effect = lambda step_id: order.append(("acquire", int(step_id)))
        state = ResidualRuntimeState()
        state.physical_plan_store = store
        consumer = _DrainProbeConsumer()

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch.object(
                residual_runtime,
                "_active_dispatch_consumers",
                return_value=[consumer],
            ),
        ):
            residual_runtime.prepare_physical_workgraph_step(7)

        self.assertEqual(
            order,
            [("drain", 7), ("acquire", 7), ("consumer_prepare", 7)],
        )

    def test_attachment_without_executor_descriptor_remains_shadow_only(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=8))
        _specialize(consumer, 2)
        state = ResidualRuntimeState()
        _activate_concrete(state, consumer)
        snapshot = state.workgraph_snapshot
        self.assertIsNotNone(snapshot)
        assert snapshot is not None

        with (
            mock.patch.object(residual_runtime, "RUNTIME", state),
            mock.patch.object(
                residual_runtime,
                "_active_dispatch_consumers",
                return_value=[consumer],
            ),
            mock.patch.object(
                consumer,
                "workgraph_region_executors",
                return_value=(),
            ),
            mock.patch.object(consumer, "activate_workgraph_regions") as activate,
        ):
            residual_runtime._activate_physical_capture_regions(snapshot)

        activate.assert_called_once_with(())

    def test_concrete_shadow_lowers_to_a_physical_plan_resources_snapshot(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=8))
        _specialize(consumer, 2)
        state = ResidualRuntimeState()

        dispatch_plan, execution_plan = _activate_concrete(state, consumer)

        physical_store = getattr(state, "physical_plan_store", None)
        self.assertIsInstance(physical_store, PlanSnapshotStore)
        assert isinstance(physical_store, PlanSnapshotStore)
        resources = physical_store.current
        self.assertIsInstance(resources, PlanResources)
        self.assertIsInstance(resources.plan, PhysicalFlowPlan)
        self.assertIsNotNone(state.workgraph_snapshot)
        assert state.workgraph_snapshot is not None
        self.assertEqual(resources.plan.plan_epoch, state.workgraph_snapshot.epoch)
        self.assertIs(state.dispatch_plan, dispatch_plan)
        self.assertIs(state.execution_plan, execution_plan)

    def test_staged_physical_snapshot_changes_only_at_the_next_prepare_safe_point(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=8))
        _specialize(consumer, 2)
        state = ResidualRuntimeState()
        _activate_concrete(state, consumer)

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            physical_store = getattr(state, "physical_plan_store", None)
            self.assertIsInstance(physical_store, PlanSnapshotStore)
            assert isinstance(physical_store, PlanSnapshotStore)
            initial = physical_store.current
            active_before_refresh = physical_store.prepare_step(40)
            self.assertIs(active_before_refresh, initial)

            _specialize(consumer, 3)
            residual_runtime.refresh_workgraph_capture_metadata()

            # Recompiling the cold snapshot stages a replacement, but cannot
            # change the resources already acquired by step 40.
            self.assertIs(physical_store.current, initial)
            self.assertIs(physical_store.prepare_step(40), initial)

            physical_store.finish_step(40)
            published = physical_store.prepare_step(41)
            self.assertIsNot(published, initial)
            self.assertGreater(published.plan.plan_epoch, initial.plan.plan_epoch)

    def test_repeating_identical_concrete_snapshot_does_not_advance_physical_epoch(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=8))
        _specialize(consumer, 2)
        state = ResidualRuntimeState()
        _activate_concrete(state, consumer)

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            physical_store = getattr(state, "physical_plan_store", None)
            self.assertIsInstance(physical_store, PlanSnapshotStore)
            assert isinstance(physical_store, PlanSnapshotStore)
            initial = physical_store.current
            initial_epoch = initial.plan.plan_epoch
            snapshot = state.workgraph_snapshot

            residual_runtime.refresh_workgraph_capture_metadata()

            self.assertIs(state.workgraph_snapshot, snapshot)
            self.assertIs(physical_store.current, initial)
            self.assertEqual(physical_store.current.plan.plan_epoch, initial_epoch)

    def test_outer_esamp_remains_legacy_eager_and_authoritative_plans_are_not_replaced(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=8))
        _specialize(consumer, 2)
        state = ResidualRuntimeState()
        dispatch_plan, execution_plan = _activate_concrete(state, consumer)

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            physical_store = getattr(state, "physical_plan_store", None)
            self.assertIsInstance(physical_store, PlanSnapshotStore)
            assert isinstance(physical_store, PlanSnapshotStore)
            physical = physical_store.current.plan
            self.assertIsInstance(physical, PhysicalFlowPlan)
            node = next(
                node
                for node in physical.nodes
                if node.node_key == NodeKey.consumer_flow(consumer.consumer_id, 0)
            )
            self.assertEqual(node.capture_policy, CapturePolicy.DISABLED)
            self.assertEqual(node.fallback_policy, FallbackPolicy.EAGER)

            _specialize(consumer, 3)
            residual_runtime.refresh_workgraph_capture_metadata()

            self.assertIs(state.dispatch_plan, dispatch_plan)
            self.assertIs(state.execution_plan, execution_plan)


if __name__ == "__main__":
    unittest.main()

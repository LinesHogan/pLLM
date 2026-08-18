#!/usr/bin/env python3
"""CPU contracts for compiler-selected fusion group execution."""

from __future__ import annotations

import unittest
from collections.abc import Mapping

import torch

from tllm.consumers.base import BaseConsumer
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.base import ConsumerFlow, FlowFusion
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.fusion import FusionExecutor
from tllm.runtime.graph_policy import GraphMode
from tllm.runtime.stats import PlacementStats


class _CapturedGroupHandle:
    def __init__(self) -> None:
        self.executions = 0
        self.replays = 0
        self.bundle_flow_ids: tuple[str, ...] = ()

    def execute(
        self,
        bundles: Mapping[str, PortBundle],
        ctx: RuntimeContext,
    ) -> None:
        del ctx
        self.executions += 1
        self.bundle_flow_ids = tuple(bundles)
        self.replays += 1

    def replay_count(self) -> int:
        return self.replays


class _EagerGroupHandle(_CapturedGroupHandle):
    def execute(
        self,
        bundles: Mapping[str, PortBundle],
        ctx: RuntimeContext,
    ) -> None:
        del ctx
        self.executions += 1
        self.bundle_flow_ids = tuple(bundles)


class _CapturedConsumer(BaseConsumer[object]):
    config = object()
    consumer_id = "captured"

    def __init__(self) -> None:
        self.consumed = 0
        self.step_ends = 0
        self.handle = _CapturedGroupHandle()

    def flows(self):
        flow = ConsumerFlow(
            reads=(),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
            max_bundle_rows=8,
            fusion=FlowFusion.SIDE_STREAM_CUDAGRAPH,
        )
        return (
            flow,
            ConsumerFlow(
                reads=(),
                writes=(),
                window="out_of_band",
                bundle_key=("engine_step_id", "phase"),
                delivery="device_lease",
                ownership="runtime_lease",
                max_bundle_rows=8,
                fusion=FlowFusion.SIDE_STREAM_CUDAGRAPH,
            ),
        )

    def consume_bundle(self, bundle, ctx) -> None:
        self.consumed += 1

    def on_step_end(self, ctx) -> None:
        self.step_ends += 1

    def fusion_group_handle(self, flow_ids: tuple[str, ...]):
        if flow_ids == ("captured#0", "captured#1"):
            return self.handle
        return None


class _NoHandleConsumer(_CapturedConsumer):
    consumer_id = "no_handle"

    def fusion_group_handle(self, flow_ids: tuple[str, ...]):
        del flow_ids
        return None


def _bundle(*, step_id: int = 1) -> PortBundle:
    return PortBundle(
        key=BundleKey(
            engine_step_id=step_id,
            phase="decode",
            request_id="",
            sample_idx=0,
        ),
        entries={},
        version=step_id,
    )


def _context() -> RuntimeContext:
    return RuntimeContext(
        runner=None,
        model=None,
        device=torch.device("cpu"),
        main_stream=None,
        is_compiling=False,
        uses_cudagraph=True,
        event_name="sampler.pre_draw",
    )


class FusionExecutorUnitTest(unittest.TestCase):
    def test_compiler_group_executes_one_group_handle_without_member_consumes(self) -> None:
        consumer = _CapturedConsumer()
        plan = DispatchPlan.build([consumer])
        plan.ensure_graph_mode(GraphMode.PIECEWISE)
        self.assertEqual(plan.fusion_groups, (("captured#0", "captured#1"),))

        executor = FusionExecutor()
        executor.configure(plan)
        stats = PlacementStats()
        for target in plan.flow_targets():
            handled = executor.execute_if_fused(
                target=target,
                bundle=_bundle(),
                context=_context(),
                stats=stats,
            )
            if not handled:
                target.consumer.consume_bundle(_bundle(), _context())

        self.assertEqual(consumer.handle.executions, 1)
        self.assertEqual(consumer.handle.bundle_flow_ids, ("captured#0", "captured#1"))
        self.assertEqual(consumer.consumed, 0)
        self.assertEqual(consumer.step_ends, 0)
        self.assertEqual(stats.fusion_execution_count, 1)
        self.assertEqual(stats.fusion_replay_count, 1)
        self.assertEqual(stats.fusion_eager_fallback_count, 0)

    def test_missing_group_handle_explicitly_falls_back_per_member(self) -> None:
        consumer = _NoHandleConsumer()
        plan = DispatchPlan.build([consumer])
        plan.ensure_graph_mode(GraphMode.PIECEWISE)
        executor = FusionExecutor()
        executor.configure(plan)
        stats = PlacementStats()

        for target in plan.flow_targets():
            handled = executor.execute_if_fused(
                target=target,
                bundle=_bundle(),
                context=_context(),
                stats=stats,
            )
            if not handled:
                target.consumer.consume_bundle(_bundle(), _context())

        self.assertEqual(consumer.handle.executions, 0)
        self.assertEqual(consumer.consumed, 2)
        self.assertEqual(stats.fusion_execution_count, 0)
        self.assertEqual(stats.fusion_replay_count, 0)
        self.assertEqual(stats.fusion_eager_fallback_count, 2)

    def test_group_execution_without_replay_is_counted_as_eager_fallback(self) -> None:
        consumer = _CapturedConsumer()
        consumer.handle = _EagerGroupHandle()
        plan = DispatchPlan.build([consumer])
        plan.ensure_graph_mode(GraphMode.PIECEWISE)
        executor = FusionExecutor()
        executor.configure(plan)
        stats = PlacementStats()

        for target in plan.flow_targets():
            self.assertTrue(
                executor.execute_if_fused(
                    target=target,
                    bundle=_bundle(),
                    context=_context(),
                    stats=stats,
                )
            )

        self.assertEqual(consumer.handle.executions, 1)
        self.assertEqual(stats.fusion_execution_count, 1)
        self.assertEqual(stats.fusion_replay_count, 0)
        self.assertEqual(stats.fusion_eager_fallback_count, 1)


if __name__ == "__main__":
    unittest.main()

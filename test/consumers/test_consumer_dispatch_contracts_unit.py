#!/usr/bin/env python3
"""Unit tests for producer-consumer contracts and runtime dispatch plan."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import List

from tllm.contracts.hidden_batch import HiddenBatch
from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.contracts.subscription import ConsumerSubscription
from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow, Locator, PortKind, PortRead
from tllm.runtime.dispatch_plan import DispatchPlan


@dataclass
class _DummyConsumer(BaseConsumer):
    cid: str
    subs: List[ConsumerSubscription]

    @property
    def consumer_id(self) -> str:
        return self.cid

    def subscriptions(self) -> List[ConsumerSubscription]:
        return list(self.subs)

    def consume(self, batch: HiddenBatch, ctx: RuntimeContext) -> None:
        _ = (batch, ctx)

    def on_tick(self, event_name: str, ctx: RuntimeContext) -> None:
        _ = (event_name, ctx)

    def on_step_end(self, ctx: RuntimeContext) -> None:
        _ = ctx


@dataclass
class _FlowConsumer(BaseConsumer):
    cid: str
    seen_bundles: List[PortBundle]

    @property
    def consumer_id(self) -> str:
        return self.cid

    def flows(self):
        return [
            ConsumerFlow(
                reads=(PortRead(kind=PortKind.REQUEST_META, locator=Locator()),),
                writes=(),
                window="background",
            )
        ]

    def consume(self, batch: HiddenBatch, ctx: RuntimeContext) -> None:
        _ = (batch, ctx)

    def consume_bundle(self, bundle: PortBundle, ctx: RuntimeContext) -> None:
        self.seen_bundles.append(bundle)

    def on_tick(self, event_name: str, ctx: RuntimeContext) -> None:
        _ = (event_name, ctx)

    def on_step_end(self, ctx: RuntimeContext) -> None:
        _ = ctx


class ConsumerDispatchContractsUnitTest(unittest.TestCase):
    def test_dispatch_plan_filters_event_phase_layer_and_capture_policy(self) -> None:
        c1 = _DummyConsumer(
            cid="c1",
            subs=[
                ConsumerSubscription(
                    consumer_id="c1",
                    event_name="layer.post",
                    phase_filter="decode",
                    layer_filter="model.model.layers[0]",
                    capture_policy="prefer",
                    dispatch_mode="inline",
                )
            ],
        )
        c2 = _DummyConsumer(
            cid="c2",
            subs=[
                ConsumerSubscription(
                    consumer_id="c2",
                    event_name="layer.post",
                    phase_filter="prefill",
                    layer_filter=None,
                    capture_policy="never",
                    dispatch_mode="consumer_async",
                )
            ],
        )

        plan = DispatchPlan.build([c1, c2])

        selected = plan.select(
            event_name="layer.post",
            phase="decode",
            layer_path="model.model.layers[0]",
            capture_enabled=True,
        )
        self.assertEqual([x.consumer.consumer_id for x in selected], ["c1"])

        selected_no_capture = plan.select(
            event_name="layer.post",
            phase="prefill",
            layer_path="model.model.layers[3]",
            capture_enabled=False,
        )
        self.assertEqual([x.consumer.consumer_id for x in selected_no_capture], ["c2"])

    def test_dispatch_plan_keeps_flow_targets_separate_from_event_targets(self) -> None:
        flow_consumer = _FlowConsumer(cid="flow", seen_bundles=[])
        plan = DispatchPlan.build([flow_consumer])

        self.assertEqual(plan.select(event_name="layer.post", phase="decode", layer_path="x", capture_enabled=True), [])
        flow_targets = plan.flow_targets()
        self.assertEqual(len(flow_targets), 1)
        self.assertIs(flow_targets[0].consumer, flow_consumer)
        self.assertEqual(flow_targets[0].flow.window, "background")


if __name__ == "__main__":
    unittest.main()

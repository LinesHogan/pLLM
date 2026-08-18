#!/usr/bin/env python3
"""End-to-end CPU contract for placement-driven deferred flow dispatch."""

from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest import mock

import torch

from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow, TimingContract
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualLocator, ResidualStream
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.ports.residual_bundle_dispatch import (
    dispatch_decode_port_bundles,
    drain_deferred_port_bundles,
)
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.vllm_patch import common_hooks as _common_hooks


class _MovableConsumer(BaseConsumer[object]):
    config = object()

    def __init__(self) -> None:
        self.seen = []
        self.step_end_count = 0

    @property
    def consumer_id(self) -> str:
        return "test.movable"

    def flows(self):
        return (
            ConsumerFlow(
                reads=(
                    ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                    RequestMeta.read(),
                ),
                writes=(),
                window="same_step",
                bundle_key=("engine_step_id", "phase"),
                timing=TimingContract(
                    mobility="deferrable",
                    deadline="next_prepare",
                    max_defer_steps=1,
                    observes_version="at_declared_window",
                ),
            ),
        )

    def consume_bundle(self, bundle, ctx) -> None:
        self.seen.append((bundle, ctx))

    def on_step_end(self, ctx) -> None:
        _ = ctx
        self.step_end_count += 1


class DeferredFlowDispatchUnitTest(unittest.TestCase):
    def _core(self, consumer: _MovableConsumer):
        runtime = ResidualRuntimeState()
        runtime.event_step_id = 7
        runtime.decode_count = 1
        runtime.decode_prompt_idxs = [3]
        runtime.decode_sample_idxs = [0]
        runtime.decode_request_ids = ["r"]
        runtime.tap_decode_hidden = {
            "layers.0": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        }
        runtime.residual_bindings = {
            "layers.0": ResidualPathBinding(
                locator=ResidualLocator(layer=0, site="block_output", phase="decode"),
                resolved_path="layers.0",
                include_request_meta=True,
            )
        }
        runtime.dispatch_plan = DispatchPlan.build([consumer])
        return type("Core", (), {"RUNTIME": runtime})()

    @staticmethod
    def _runner():
        return type(
            "Runner",
            (),
            {
                "device": torch.device("cpu"),
                "model": None,
                "use_cuda_graph": False,
                "compilation_config": None,
            },
        )()

    def test_compiled_placement_enqueues_then_sampler_bubble_drains(self) -> None:
        consumer = _MovableConsumer()
        core = self._core(consumer)

        self.assertTrue(core.RUNTIME.dispatch_plan.has_movable_flow_targets())
        self.assertEqual(dispatch_decode_port_bundles(core=core, runner=self._runner()), 1)
        self.assertEqual(consumer.seen, [])
        self.assertEqual(core.RUNTIME.deferred_work_queue.depth, 1)

        drained = drain_deferred_port_bundles(
            runtime=core.RUNTIME,
            step_id=7,
            hook="sampler.pre_draw",
        )

        self.assertEqual(drained, 1)
        self.assertEqual(len(consumer.seen), 1)
        self.assertEqual(consumer.seen[0][0].version, 7)
        self.assertEqual(consumer.step_end_count, 1)

    def test_cross_check_detects_borrowed_tensor_mutation(self) -> None:
        consumer = _MovableConsumer()
        core = self._core(consumer)
        with mock.patch.dict(os.environ, {"TLLM_DEBUG_TIMING_MODE": "cross_check"}):
            dispatch_decode_port_bundles(core=core, runner=self._runner())
            core.RUNTIME.tap_decode_hidden["layers.0"].fill_(9.0)
            with self.assertRaisesRegex(RuntimeError, "timing cross-check failed"):
                drain_deferred_port_bundles(
                    runtime=core.RUNTIME,
                    step_id=7,
                    hook="sampler.pre_draw",
                )

    def test_runtime_event_drains_work_at_compiler_selected_hook(self) -> None:
        consumer = _MovableConsumer()
        core = self._core(consumer)
        dispatch_decode_port_bundles(core=core, runner=self._runner())
        queued = next(iter(core.RUNTIME.deferred_work_queue))
        core.RUNTIME.deferred_work_queue.clear()
        core.RUNTIME.deferred_work_queue.enqueue(
            replace(queued, eligible_hook="sample_tokens.post")
        )

        self.assertEqual(
            _common_hooks.dispatch_runtime_event(
                runtime=core.RUNTIME,
                runner=self._runner(),
                event_name="sample_tokens.pre",
                phase="decode",
            ),
            0,
        )
        self.assertEqual(consumer.seen, [])

        self.assertEqual(
            _common_hooks.dispatch_runtime_event(
                runtime=core.RUNTIME,
                runner=self._runner(),
                event_name="sample_tokens.post",
                phase="decode",
            ),
            0,
        )
        self.assertEqual(len(consumer.seen), 1)
        self.assertFalse(core.RUNTIME.deferred_work_queue)


if __name__ == "__main__":
    unittest.main()

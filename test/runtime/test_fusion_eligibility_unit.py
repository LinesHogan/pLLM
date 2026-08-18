#!/usr/bin/env python3
"""Unit contracts for B4 fusion eligibility and fallback metadata."""

from __future__ import annotations

import unittest

from tllm.core import compile_placement_schedule
from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow, PortKind, PortWrite
from tllm.ports.residual_stream import ResidualStream
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.graph_policy import GraphMode
from tllm.runtime.placement import PlacementFlowDesc, PlacementWriteDesc


def _descriptor(
    *,
    flow_id: str = "fusion#0",
    declared_hook: str = "layer.post",
    delivery: str = "device_lease",
    rows: int = 8,
    port: str = "residual:x",
    fusion_candidate: bool = True,
) -> PlacementFlowDesc:
    return PlacementFlowDesc(
        flow_id=flow_id,
        consumer_id="fusion",
        declared_hook=declared_hook,
        mobility="pinned",
        deadline="step_end",
        max_defer_steps=0,
        observes_version="at_declared_window",
        reads=(),
        writes=(
            PlacementWriteDesc(
                port=port,
                order=0,
                mode="add",
                fusion_candidate=fusion_candidate,
            ),
        ),
        delivery=delivery,
        ownership="runtime_lease",
        max_bundle_rows=rows,
        effect_hook="layer.post",
    )


class FusionEligibilityUnitTest(unittest.TestCase):
    def test_public_contract_rejects_unstable_or_dynamic_candidate(self) -> None:
        write = ResidualStream.write(
            layer=0,
            site="block_output",
            phase="decode",
            mode="add",
            fusion_candidate=True,
        )
        with self.assertRaisesRegex(ValueError, "stable device_lease"):
            ConsumerFlow(reads=(), writes=(write,), window="same_step")
        with self.assertRaisesRegex(ValueError, "max_bundle_rows"):
            ConsumerFlow(
                reads=(),
                writes=(write,),
                window="same_step",
                bundle_key=("engine_step_id", "phase"),
                delivery="device_lease",
                ownership="runtime_lease",
            )

    def test_public_write_contract_rejects_unsupported_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "replace.*add"):
            PortWrite(
                kind=PortKind.CPU_EXPORT,
                mode="append",
                fusion_candidate=True,
            )

    def test_piecewise_emits_group_while_eager_emits_fallback(self) -> None:
        piecewise = compile_placement_schedule([_descriptor()], "piecewise")
        eager = compile_placement_schedule([_descriptor()], "eager")

        self.assertEqual(piecewise.fusion_groups, [["fusion#0"]])
        self.assertEqual(piecewise.fusion_rejections, [])
        self.assertEqual(eager.fusion_groups, [])
        self.assertEqual(eager.fusion_rejections[0][0], "fusion#0")
        self.assertIn("eager", eager.fusion_rejections[0][1])

    def test_compiler_defense_rejects_dynamic_shape_candidate(self) -> None:
        schedule = compile_placement_schedule([_descriptor(rows=0)], "full")

        self.assertEqual(schedule.fusion_groups, [])
        self.assertIn("static", schedule.fusion_rejections[0][1])

    def test_effect_hook_is_independent_from_host_flow_placement(self) -> None:
        schedule = compile_placement_schedule(
            [_descriptor(declared_hook="execute_model.post")],
            "piecewise",
        )

        self.assertEqual(schedule.placements, [("fusion#0", "execute_model.post")])
        self.assertEqual(schedule.fusion_groups, [["fusion#0"]])
        self.assertEqual(schedule.fusion_rejections, [])

    def test_non_fusion_node_splits_topologically_contiguous_regions(self) -> None:
        descriptors = [
            _descriptor(flow_id="fusion#0", port="residual:a"),
            _descriptor(flow_id="barrier#0", port="residual:b", fusion_candidate=False),
            _descriptor(flow_id="fusion#1", port="residual:c"),
        ]

        schedule = compile_placement_schedule(descriptors, "piecewise")

        self.assertEqual(schedule.fusion_groups, [["fusion#0"], ["fusion#1"]])

    def test_initial_eager_fallback_metadata_is_published_once(self) -> None:
        write = ResidualStream.write(
            layer=0,
            site="block_output",
            phase="decode",
            mode="add",
            fusion_candidate=True,
        )

        class FusionConsumer(BaseConsumer[object]):
            config = object()
            consumer_id = "fusion"

            def flows(self):
                return (
                    ConsumerFlow(
                        reads=(),
                        writes=(write,),
                        window="same_step",
                        bundle_key=("engine_step_id", "phase"),
                        delivery="device_lease",
                        ownership="runtime_lease",
                        max_bundle_rows=8,
                    ),
                )

        plan = DispatchPlan.build([FusionConsumer()])

        self.assertEqual(len(plan.fusion_rejections), 1)
        self.assertTrue(plan.ensure_graph_mode(GraphMode.EAGER))
        self.assertFalse(plan.ensure_graph_mode(GraphMode.EAGER))

    def test_production_lowering_keeps_host_dispatch_outside_capture_and_fuses_effect(self) -> None:
        write = ResidualStream.write(
            layer=0,
            site="block_output",
            phase="decode",
            mode="add",
            fusion_candidate=True,
        )

        class FusionConsumer(BaseConsumer[object]):
            config = object()
            consumer_id = "fusion.production"

            def flows(self):
                return (
                    ConsumerFlow(
                        reads=(),
                        writes=(write,),
                        window="same_step",
                        bundle_key=("engine_step_id", "phase"),
                        delivery="device_lease",
                        ownership="runtime_lease",
                        max_bundle_rows=8,
                    ),
                )

        plan = DispatchPlan.build([FusionConsumer()])
        plan.ensure_graph_mode(GraphMode.PIECEWISE)

        target = plan.flow_targets()[0]
        self.assertEqual(target.placement_hook, "execute_model.post")
        self.assertEqual(plan.fusion_groups, (("fusion.production#0",),))
        self.assertEqual(plan.fusion_rejections, ())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""B5 full-hook DAG list-scheduling contracts."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, replace

from tllm.core import compile_placement_schedule
from tllm.ports.base import ConsumerFlow, Locator, PortKind, PortWrite, TimingContract
from tllm.ports.residual_stream import ResidualStream
from tllm.runtime.placement import (
    PlacementFlowDesc,
    PlacementHookCost,
    PlacementResourceWriteDesc,
    PlacementWriteDesc,
    VersionedStateResourceDesc,
    compile_flow_placement,
    port_address_desc,
)


@dataclass(frozen=True)
class _CollisionLocator(Locator):
    a: str
    b: str


def _flow(
    flow_id: str,
    *,
    declared: str,
    mobility: str,
    reads: tuple[str, ...] = (),
    writes: tuple[PlacementWriteDesc, ...] = (),
    consumer: str = "",
    earliest: str | None = None,
) -> PlacementFlowDesc:
    movable = mobility in ("deferrable", "free")
    return PlacementFlowDesc(
        flow_id=flow_id,
        consumer_id=consumer,
        declared_hook=declared,
        mobility=mobility,
        deadline="next_prepare" if movable else "step_end",
        max_defer_steps=1 if movable else 0,
        observes_version="latest_at_execution" if mobility != "pinned" else "at_declared_window",
        reads=reads,
        writes=writes,
        earliest_hook=earliest,
        allowed_hooks=(),
    )


class FullPlacementCompilerUnitTest(unittest.TestCase):
    def test_optimizer_parameter_write_is_placed_after_sampler_version_read(self) -> None:
        params_v0 = VersionedStateResourceDesc(
            kind="model_parameters",
            namespace="esamp.shared",
            name="distiller",
            version=0,
        )
        optimizer = replace(
            _flow(
                "optimizer",
                declared="execute_model.post",
                mobility="deferrable",
            ),
            resource_writes=(
                PlacementResourceWriteDesc(replace(params_v0, version=1)),
            ),
        )
        sampler = replace(
            _flow(
                "sampler",
                declared="sampler.pre_draw",
                mobility="pinned",
            ),
            resource_reads=(params_v0,),
        )

        schedule = compile_placement_schedule([optimizer, sampler], "eager")

        self.assertEqual(schedule.execution_order, ["sampler", "optimizer"])
        self.assertEqual(schedule.placements[0], ("optimizer", "sampler.pre_draw"))

    def test_step_lifetime_device_lease_moves_after_sampler_parameter_reads(self) -> None:
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(
                    layer=0,
                    site="block_output",
                    phase="decode",
                    role="source",
                ),
            ),
            writes=(),
            window="out_of_band",
            delivery="device_lease",
            ownership="runtime_lease",
            lease_lifetime="step",
            bundle_key=("engine_step_id", "phase"),
            timing=TimingContract(
                mobility="deferrable",
                deadline="next_prepare",
                max_defer_steps=1,
                observes_version="at_declared_window",
            ),
        )

        schedule = compile_flow_placement([("lease#0", flow)])

        self.assertEqual(schedule.by_flow_id, {"lease#0": "sampler.pre_draw"})

    def test_consume_call_device_lease_cannot_move(self) -> None:
        with self.assertRaisesRegex(ValueError, "lease_lifetime='step'"):
            ConsumerFlow(
                reads=(
                    ResidualStream.read(
                        layer=0,
                        site="block_output",
                        phase="decode",
                    ),
                ),
                writes=(),
                window="out_of_band",
                delivery="device_lease",
                ownership="runtime_lease",
                bundle_key=("engine_step_id", "phase"),
                timing=TimingContract(
                    mobility="deferrable",
                    deadline="next_prepare",
                    max_defer_steps=1,
                    observes_version="at_declared_window",
                ),
            )

    def test_structural_port_addresses_do_not_collide_on_delimiter_text(self) -> None:
        first = port_address_desc(
            PortWrite(
                kind=PortKind.CPU_EXPORT,
                locator=_CollisionLocator(a="x,b=y", b=""),
            )
        )
        second = port_address_desc(
            PortWrite(
                kind=PortKind.CPU_EXPORT,
                locator=_CollisionLocator(a="x", b="y,b="),
            )
        )

        self.assertNotEqual(first, second)
        flows = [
            _flow(
                "first",
                declared="execute_model.post",
                mobility="pinned",
                writes=(PlacementWriteDesc(first, 0, "replace", False),),
            ),
            _flow(
                "second",
                declared="execute_model.post",
                mobility="pinned",
                writes=(PlacementWriteDesc(second, 0, "replace", False),),
            ),
        ]

        schedule = compile_placement_schedule(flows, "eager")

        self.assertEqual(set(schedule.execution_order), {"first", "second"})

    def test_advanceable_uses_profiled_earlier_bubble(self) -> None:
        flow = _flow(
            "advance",
            declared="execute_model.post",
            mobility="advanceable",
            earliest="prepare_inputs.post",
        )
        costs = [
            PlacementHookCost(hook="prepare_inputs.post", bubble_score=500),
            PlacementHookCost(hook="execute_model.post", bubble_score=0),
        ]

        schedule = compile_placement_schedule([flow], "eager", costs)

        self.assertEqual(schedule.placements, [("advance", "prepare_inputs.post")])

    def test_dependency_bound_prevents_moving_past_pinned_successor(self) -> None:
        write = PlacementWriteDesc(port="x", order=0, mode="replace", fusion_candidate=False)
        flows = [
            _flow(
                "producer",
                declared="execute_model.post",
                mobility="deferrable",
                writes=(write,),
            ),
            _flow(
                "reader",
                declared="execute_model.post",
                mobility="pinned",
                reads=("x",),
            ),
        ]

        schedule = compile_placement_schedule(flows, "eager")

        self.assertEqual(schedule.placements[0], ("producer", "execute_model.post"))

    def test_reversed_waw_order_forms_cycle_with_state_order(self) -> None:
        flows = [
            _flow(
                "first",
                declared="execute_model.post",
                mobility="pinned",
                consumer="stateful",
                writes=(PlacementWriteDesc("x", 2, "replace", False),),
            ),
            _flow(
                "second",
                declared="execute_model.post",
                mobility="pinned",
                consumer="stateful",
                writes=(PlacementWriteDesc("x", 1, "replace", False),),
            ),
        ]

        with self.assertRaisesRegex(RuntimeError, "dependency cycle"):
            compile_placement_schedule(flows, "eager")

    def test_opposite_waw_orders_across_shared_ports_form_cycle(self) -> None:
        flows = [
            _flow(
                "a",
                declared="execute_model.post",
                mobility="pinned",
                writes=(
                    PlacementWriteDesc("x", 0, "replace", False),
                    PlacementWriteDesc("y", 2, "replace", False),
                ),
            ),
            _flow(
                "b",
                declared="execute_model.post",
                mobility="pinned",
                writes=(
                    PlacementWriteDesc("x", 1, "replace", False),
                    PlacementWriteDesc("y", 1, "replace", False),
                ),
            ),
        ]

        with self.assertRaisesRegex(RuntimeError, "dependency cycle"):
            compile_placement_schedule(flows, "eager")

    def test_duplicate_waw_order_on_second_shared_port_is_rejected(self) -> None:
        flows = [
            _flow(
                "a",
                declared="execute_model.post",
                mobility="pinned",
                writes=(
                    PlacementWriteDesc("x", 0, "replace", False),
                    PlacementWriteDesc("y", 3, "replace", False),
                ),
            ),
            _flow(
                "b",
                declared="execute_model.post",
                mobility="pinned",
                writes=(
                    PlacementWriteDesc("x", 1, "replace", False),
                    PlacementWriteDesc("y", 3, "replace", False),
                ),
            ),
        ]

        with self.assertRaisesRegex(RuntimeError, "write port 'y' with duplicate order 3"):
            compile_placement_schedule(flows, "eager")

    def test_consistent_multi_port_waw_order_has_one_topological_direction(self) -> None:
        flows = [
            _flow(
                "later",
                declared="execute_model.post",
                mobility="pinned",
                writes=(
                    PlacementWriteDesc("x", 2, "replace", False),
                    PlacementWriteDesc("y", 4, "replace", False),
                ),
            ),
            _flow(
                "earlier",
                declared="execute_model.post",
                mobility="pinned",
                writes=(
                    PlacementWriteDesc("x", 1, "replace", False),
                    PlacementWriteDesc("y", 3, "replace", False),
                ),
            ),
        ]

        schedule = compile_placement_schedule(flows, "eager")

        self.assertEqual(schedule.execution_order, ["earlier", "later"])

    def test_backend_allowed_hooks_fail_fast_when_domain_is_empty(self) -> None:
        flow = _flow(
            "unsupported",
            declared="execute_model.post",
            mobility="deferrable",
        )
        flow = replace(flow, allowed_hooks=("prepare_inputs.pre",))

        with self.assertRaisesRegex(RuntimeError, "no legal placement"):
            compile_placement_schedule([flow], "eager")


if __name__ == "__main__":
    unittest.main()

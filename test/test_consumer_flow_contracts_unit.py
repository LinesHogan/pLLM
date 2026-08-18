#!/usr/bin/env python3
"""Unit tests for public consumer flow contracts."""

from __future__ import annotations

import unittest

from tllm.ports.base import (
    ConsumerFlow,
    FlowDelivery,
    FlowWindow,
    Locator,
    PortKind,
    PortRead,
    PortWrite,
)
from tllm.ports.residual_stream import ResidualLocator


class ConsumerFlowContractsUnitTest(unittest.TestCase):
    def test_invalid_window_and_delivery_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported flow window"):
            ConsumerFlow(reads=(), writes=(), window="not_a_window")
        with self.assertRaisesRegex(ValueError, "unsupported delivery mode"):
            ConsumerFlow(reads=(), writes=(), window="background", delivery="not_a_delivery")

    def test_device_lease_delivery_requires_runtime_lease_ownership(self) -> None:
        with self.assertRaisesRegex(ValueError, "device_lease delivery requires ownership"):
            ConsumerFlow(
                reads=(),
                writes=(),
                window="background",
                delivery="device_lease",
            )

    def test_consumer_flow_accepts_public_enum_values(self) -> None:
        flow = ConsumerFlow(
            reads=(),
            writes=(),
            window=FlowWindow.OUT_OF_BAND,
            delivery=FlowDelivery.BUNDLE,
        )

        self.assertEqual(str(flow.window), "out_of_band")
        self.assertEqual(str(flow.delivery), "bundle")

    def test_port_write_rejects_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported write mode"):
            PortWrite(kind=PortKind.CPU_EXPORT, locator=Locator(), mode="nope")

    def test_bundle_fields_must_be_unique_after_role_resolution(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate bundle field `residual_stream`"):
            ConsumerFlow(
                reads=(
                    PortRead(kind=PortKind.RESIDUAL_STREAM),
                    PortRead(kind=PortKind.RESIDUAL_STREAM),
                ),
                writes=(),
                window="same_step",
            )

    def test_explicit_roles_disambiguate_reads_of_the_same_kind(self) -> None:
        flow = ConsumerFlow(
            reads=(
                PortRead(
                    kind=PortKind.RESIDUAL_STREAM,
                    locator=ResidualLocator(layer=0, site="block_output", phase="decode"),
                    role="source",
                ),
                PortRead(
                    kind=PortKind.RESIDUAL_STREAM,
                    locator=ResidualLocator(layer=1, site="block_output", phase="decode"),
                    role="target",
                ),
            ),
            writes=(),
            window="same_step",
        )

        self.assertEqual(flow.bundle_fields, ("source", "target"))

    def test_bundle_key_fields_are_validated_at_declaration_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown bundle_key field `engine_setp_id`"):
            ConsumerFlow(
                reads=(),
                writes=(),
                window="same_step",
                bundle_key=("engine_setp_id", "phase"),
            )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit tests for the timing-contract IR (B1)."""

from __future__ import annotations

import unittest

from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.consumers import registry as consumer_registry
from tllm.ports.base import (
    ConsumerFlow,
    TimingContract,
    TimingDeadline,
    TimingMobility,
    VersionObservation,
)
from tllm.ports.request_meta import RequestMeta
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.ports.frame import Ownership, PortFrame
from tllm.ports.base import PortKind


class TimingContractUnitTest(unittest.TestCase):
    def test_defaults_are_pinned(self) -> None:
        contract = TimingContract()
        self.assertEqual(contract.mobility, TimingMobility.PINNED)
        self.assertEqual(contract.max_defer_steps, 0)

    def test_string_normalization(self) -> None:
        contract = TimingContract(mobility="deferrable", deadline="next_prepare", max_defer_steps=2, observes_version="stale_ok")
        self.assertEqual(contract.mobility, TimingMobility.DEFERRABLE)
        self.assertEqual(contract.deadline, TimingDeadline.NEXT_PREPARE)
        self.assertEqual(contract.observes_version, VersionObservation.STALE_OK)

    def test_invalid_combinations_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_defer_steps=0"):
            TimingContract(mobility="pinned", max_defer_steps=1)
        with self.assertRaisesRegex(ValueError, "at_declared_window"):
            TimingContract(mobility="pinned", observes_version="stale_ok")
        with self.assertRaisesRegex(ValueError, ">=1"):
            TimingContract(mobility="deferrable", deadline="next_prepare")
        with self.assertRaisesRegex(ValueError, "at_declared_window"):
            TimingContract(mobility="advanceable", observes_version="at_declared_window")
        with self.assertRaisesRegex(ValueError, "step_end"):
            TimingContract(mobility="free", deadline="step_end", max_defer_steps=1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            TimingContract(max_defer_steps=-1)
        with self.assertRaisesRegex(ValueError, "mobility"):
            TimingContract(mobility="teleport")


class ConsumerFlowTimingUnitTest(unittest.TestCase):
    def _flow(self, **kwargs) -> ConsumerFlow:
        return ConsumerFlow(
            reads=(RequestMeta.read(),),
            writes=(),
            window="same_step",
            bundle_key=("engine_step_id", "phase"),
            **kwargs,
        )

    def test_default_timing_is_pinned(self) -> None:
        flow = self._flow()
        self.assertEqual(flow.timing.mobility, TimingMobility.PINNED)

    def test_string_shorthand_normalizes(self) -> None:
        flow = self._flow(timing="pinned")
        self.assertIsInstance(flow.timing, TimingContract)
        self.assertEqual(flow.timing.mobility, TimingMobility.PINNED)

    def test_movable_flow_validates(self) -> None:
        flow = self._flow(
            timing=TimingContract(mobility="deferrable", deadline="next_prepare", max_defer_steps=2, observes_version="latest_at_execution")
        )
        self.assertEqual(flow.timing.mobility, TimingMobility.DEFERRABLE)

    def test_execution_time_version_observation_fails_before_runtime_dispatch(self) -> None:
        flow = self._flow(
            timing=TimingContract(
                mobility="deferrable",
                deadline="next_prepare",
                max_defer_steps=1,
                observes_version="latest_at_execution",
            )
        )

        class UnsupportedVersionConsumer:
            consumer_id = "test.unsupported_version"

            def flows(self):
                return (flow,)

        with self.assertRaisesRegex(RuntimeError, "no execution-time port-version resolver"):
            DispatchPlan.build([UnsupportedVersionConsumer()])

    def test_movable_device_lease_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "device_lease"):
            ConsumerFlow(
                reads=(RequestMeta.read(),),
                writes=(),
                window="same_step",
                bundle_key=("engine_step_id", "phase"),
                delivery="device_lease",
                ownership="runtime_lease",
                timing=TimingContract(mobility="free", deadline="generate_end", max_defer_steps=1, observes_version="stale_ok"),
            )


class VersionFieldUnitTest(unittest.TestCase):
    def test_bundle_version_defaults_zero(self) -> None:
        bundle = PortBundle(key=BundleKey(engine_step_id=0, phase="decode", request_id="r", sample_idx=0))
        self.assertEqual(bundle.version, 0)

    def test_frame_version_defaults_zero(self) -> None:
        frame = PortFrame(
            key=BundleKey(engine_step_id=0, phase="decode", request_id="r", sample_idx=0),
            kind=PortKind.REQUEST_META,
            locator=None,
            payload=None,
            ownership=Ownership.BORROWED,
            ready_window="same_step",
        )
        self.assertEqual(frame.version, 0)


class DispatchPlanTimingValidationUnitTest(unittest.TestCase):
    """DispatchPlan.build runs flow timing contracts through the Rust compiler."""

    def test_builtin_consumers_validate(self) -> None:
        consumer = consumer_registry.build_consumer("dummy", DummyConsumerConfig())
        plan = DispatchPlan.build([consumer])
        self.assertTrue(plan.has_active_targets())

    def test_invalid_timing_fails_at_plan_build(self) -> None:
        class BadFlow:
            consumer_id = "test.bad_timing"

            def flows(self):
                # Bypass ConsumerFlow's own validation to prove the plan-build
                # gate catches it independently (defense in depth).
                flow = ConsumerFlow(
                    reads=(RequestMeta.read(),),
                    writes=(),
                    window="same_step",
                    bundle_key=("engine_step_id", "phase"),
                )
                object.__setattr__(flow, "timing", TimingContract(mobility="advanceable", observes_version="stale_ok"))
                object.__setattr__(flow.timing, "mobility", "teleport")  # invalid enum value
                return (flow,)

        with self.assertRaises(RuntimeError):
            DispatchPlan.build([BadFlow()])


if __name__ == "__main__":
    unittest.main()

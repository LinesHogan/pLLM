#!/usr/bin/env python3
"""Unit tests for the staged residual write channel."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow, PortWrite, WriteMode, WriteRows
from tllm.ports.residual_stream import ResidualLocator, ResidualStream
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.ports import residual_capture_hooks, residual_write
from tllm.runtime.ports.residual_write import ResidualWriteLease, WriteLeaseKey


def _lease(*, mode: WriteMode = WriteMode.REPLACE, order: int = 0, writer: str = "w0") -> ResidualWriteLease:
    return ResidualWriteLease(
        key=WriteLeaseKey(layer_path="layers.0", writer_id=writer, order=order),
        mode=mode,
        rows=WriteRows.DECODE_ONLY,
        max_rows=4,
        hidden_size=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


class ResidualWriteLeaseUnitTest(unittest.TestCase):
    def test_stage_copies_into_persistent_buffers(self) -> None:
        lease = _lease()
        delta = torch.full((2, 3), 7.0)
        rows = torch.tensor([1, 3])
        lease.stage(delta=delta, row_idx=rows)
        self.assertEqual(lease.count, 2)
        self.assertTrue(torch.equal(lease.delta[:2], delta))
        self.assertTrue(torch.equal(lease.row_idx[:2], rows))

    def test_stage_rejects_capacity_overflow_and_shape_mismatch(self) -> None:
        lease = _lease()
        with self.assertRaisesRegex(ValueError, "exceeds lease capacity"):
            lease.stage(delta=torch.zeros((5, 3)), row_idx=torch.arange(5))
        with self.assertRaisesRegex(ValueError, "row mismatch"):
            lease.stage(delta=torch.zeros((2, 3)), row_idx=torch.arange(3))
        with self.assertRaisesRegex(ValueError, "hidden width mismatch"):
            lease.stage(delta=torch.zeros((1, 4)), row_idx=torch.arange(1))

    def test_stage_validate_rows_rejects_duplicates(self) -> None:
        lease = _lease()
        with self.assertRaisesRegex(ValueError, "must be unique"):
            lease.stage(
                delta=torch.zeros((2, 3)),
                row_idx=torch.tensor([1, 1]),
                validate_rows=True,
            )

    def test_wrapper_mode_has_no_lease(self) -> None:
        with self.assertRaisesRegex(ValueError, "replace/add"):
            _lease(mode=WriteMode.WRAPPER)


class ApplyPendingWritesUnitTest(unittest.TestCase):
    def test_replace_applies_then_clears(self) -> None:
        lease = _lease(mode=WriteMode.REPLACE)
        lease.stage(delta=torch.full((2, 3), 9.0), row_idx=torch.tensor([0, 2]))
        tensor = torch.zeros((4, 3))
        residual_write.apply_pending_writes({lease.key: lease}, "layers.0", tensor)
        self.assertTrue(torch.equal(tensor[0], torch.full((3,), 9.0)))
        self.assertTrue(torch.equal(tensor[2], torch.full((3,), 9.0)))
        self.assertTrue(torch.equal(tensor[1], torch.zeros(3)))
        self.assertEqual(lease.count, 0)

    def test_add_accumulates(self) -> None:
        lease = _lease(mode=WriteMode.ADD)
        lease.stage(delta=torch.ones((1, 3)), row_idx=torch.tensor([1]))
        tensor = torch.full((4, 3), 5.0)
        residual_write.apply_pending_writes({lease.key: lease}, "layers.0", tensor)
        self.assertTrue(torch.equal(tensor[1], torch.full((3,), 6.0)))

    def test_multiple_writers_apply_in_order(self) -> None:
        add = _lease(mode=WriteMode.ADD, order=0, writer="adder")
        replace = _lease(mode=WriteMode.REPLACE, order=1, writer="replacer")
        add.stage(delta=torch.ones((1, 3)), row_idx=torch.tensor([0]))
        replace.stage(delta=torch.full((1, 3), 4.0), row_idx=torch.tensor([0]))
        tensor = torch.zeros((4, 3))
        leases = {add.key: add, replace.key: replace}
        residual_write.apply_pending_writes(leases, "layers.0", tensor)
        # add (order=0) runs first, replace (order=1) overwrites.
        self.assertTrue(torch.equal(tensor[0], torch.full((3,), 4.0)))

    def test_other_paths_untouched(self) -> None:
        lease = _lease()
        lease.stage(delta=torch.ones((1, 3)), row_idx=torch.tensor([0]))
        tensor = torch.zeros((4, 3))
        residual_write.apply_pending_writes({lease.key: lease}, "layers.9", tensor)
        self.assertTrue(torch.equal(tensor, torch.zeros((4, 3))))
        self.assertEqual(lease.count, 1)  # not consumed by another layer's forward


class BuildWriteLeasesUnitTest(unittest.TestCase):
    def test_unresolved_locator_fails_fast(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not resolved"):
            residual_write.build_write_leases(
                writes=[("c0", ResidualLocator(layer=0, site="block_output", phase="decode"),
                         WriteMode.REPLACE, WriteRows.DECODE_ONLY, 0)],
                resolve_path=lambda locator: None,
                device=torch.device("cpu"),
                max_rows=4,
                hidden_size=3,
                dtype=torch.float32,
            )


class _WriterConsumer(BaseConsumer):
    def __init__(self, consumer_id: str, *, order: int, mode: str = "replace") -> None:
        self._consumer_id = consumer_id
        self._order = order
        self._mode = mode

    @property
    def consumer_id(self) -> str:
        return self._consumer_id

    def flows(self):
        return (
            ConsumerFlow(
                reads=(),
                writes=(
                    ResidualStream.write(
                        layer=0,
                        site="block_output",
                        phase="decode",
                        mode=self._mode,
                        order=self._order,
                    ),
                ),
                window="out_of_band",
            ),
        )


class DispatchPlanWriteUnitTest(unittest.TestCase):
    def test_required_residual_writes(self) -> None:
        plan = DispatchPlan.build([_WriterConsumer("c0", order=0)])
        self.assertEqual(plan.required_residual_writes(), {(0, "block_output", "decode")})

    def test_same_locator_same_order_conflicts(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting residual writes"):
            DispatchPlan.build([_WriterConsumer("c0", order=0), _WriterConsumer("c1", order=0)])

    def test_same_locator_distinct_order_ok(self) -> None:
        plan = DispatchPlan.build([_WriterConsumer("c0", order=0), _WriterConsumer("c1", order=1)])
        self.assertEqual(len(plan.residual_writes()), 2)


class CompileTimeWriteValidationUnitTest(unittest.TestCase):
    def test_write_to_unknown_kind_fails_fast(self) -> None:
        from tllm.producers.registry import ExecutionPlanError, ProducerRegistry, compile_execution_plan

        class _TraceWriter(_WriterConsumer):
            def flows(self):
                return (
                    ConsumerFlow(
                        reads=(),
                        writes=(PortWrite(kind=__import__("tllm.ports.base", fromlist=["PortKind"]).PortKind.GENERATION_TRACE, mode="append"),),
                        window="out_of_band",
                    ),
                )

        with self.assertRaisesRegex(ExecutionPlanError, "no registered producer offers it"):
            compile_execution_plan(registry=ProducerRegistry(), consumers=[_TraceWriter("c0", order=0)])

    def test_residual_write_compiles_against_default_plan(self) -> None:
        from tllm.runtime.execution_plan import compile_default_plan

        plan = compile_default_plan(consumers=[_WriterConsumer("c0", order=0)])
        self.assertIsNotNone(plan)

    def test_unsupported_write_mode_fails_fast(self) -> None:
        from tllm.runtime.execution_plan import compile_default_plan
        from tllm.producers.registry import ExecutionPlanError

        consumer = _WriterConsumer("c0", order=0, mode="add")
        # ``add`` is supported; ``append`` is rejected at the port helper, so
        # bypass it to reach the compiler check.
        flow = consumer.flows()[0]
        bad_write = PortWrite(
            kind=flow.writes[0].kind,
            locator=flow.writes[0].locator,
            mode="append",
        )
        consumer._flow_override = ConsumerFlow(reads=(), writes=(bad_write,), window="out_of_band")
        consumer.flows = lambda: (consumer._flow_override,)  # type: ignore[method-assign]
        with self.assertRaisesRegex(ExecutionPlanError, "only support"):
            compile_default_plan(consumers=[consumer])


class DebugGraphModeSwitchUnitTest(unittest.TestCase):
    def test_debug_env_forces_graph_mode(self) -> None:
        from tllm.runtime import graph_policy

        runner = type("Runner", (), {"enforce_eager": True})()
        self.assertIs(graph_policy.resolve_graph_mode(runner), graph_policy.GraphMode.EAGER)
        with mock.patch.dict(os.environ, {"TLLM_DEBUG_GRAPH_MODE": "full"}):
            self.assertIs(graph_policy.resolve_graph_mode(runner), graph_policy.GraphMode.FULL)
        with mock.patch.dict(os.environ, {"TLLM_DEBUG_GRAPH_MODE": "bogus"}):
            with self.assertRaisesRegex(Exception, "TLLM_DEBUG_GRAPH_MODE"):
                graph_policy.resolve_graph_mode(runner)


class TapWriteIntegrationUnitTest(unittest.TestCase):
    def test_tap_applies_staged_write_before_capture(self) -> None:
        class _Layer(torch.nn.Module):
            def forward(self, x):
                return x + 1

        layer = _Layer()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.tensor([0, 1], dtype=torch.long)
        runtime.decode_valid_mask = torch.ones((2, 1))
        runtime.decode_count = 2
        runtime.tap_decode_hidden = {"layers.0": torch.zeros((2, 3))}
        runtime.launch_consumer_from_hooks = True
        runtime.dispatch_plan = type("Plan", (), {"has_active_targets": staticmethod(lambda: True)})()
        runtime.consumer = None
        runtime.source_resolved_path = "layers.0"
        runtime.target_resolved_path = "layers.1"

        lease = ResidualWriteLease(
            key=WriteLeaseKey(layer_path="layers.0", writer_id="c0", order=0),
            mode=WriteMode.REPLACE,
            rows=WriteRows.DECODE_ONLY,
            max_rows=2,
            hidden_size=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        runtime.write_leases = {lease.key: lease}
        lease.stage(delta=torch.full((1, 3), 42.0), row_idx=torch.tensor([1]))

        core = type("Core", (), {"RUNTIME": runtime})()
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(residual_capture_hooks._sampler_patch, "maybe_capture_source_precompute"), mock.patch.object(
            residual_capture_hooks._hidden_bridge, "dispatch_layer_lifecycle_events"
        ):
            handles = residual_capture_hooks.install_layer_forward_taps(
                core=core, runner=runner, resolved_layers={"layers.0": layer}
            )
            x = torch.zeros((2, 3))
            out = layer(x)

        # The staged write hit the live output (x+1 then row 1 replaced)...
        self.assertTrue(torch.equal(out[1], torch.full((3,), 42.0)))
        self.assertTrue(torch.equal(out[0], torch.ones(3)))
        # ...and the capture saw the written stream.
        self.assertTrue(torch.equal(runtime.tap_decode_hidden["layers.0"][1], torch.full((3,), 42.0)))
        self.assertEqual(lease.count, 0)

        # Symmetric teardown restores the original forward.
        orig = handles["layers.0"].orig_forward
        residual_capture_hooks.restore_layer_forward_taps(handles)
        self.assertEqual(handles, {})
        self.assertIs(layer.forward, orig)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Static-IR tests for declare_program (plan Phase 1).

Covers named values, DATA/CONTROL/STATE edges, deterministic scheduling,
non-fatal diagnostics (PG300/PG301), bound-method discovery, and the
regression rules: plain Python calls stay transparent and legacy
``flow(...).capture()`` is untouched.
"""

from __future__ import annotations

import unittest

import torch

from tllm.program import (
    EdgeKind,
    ProgramDecl,
    StateSpec,
    declare_program,
)
from tllm.ports.base import TimingDeadline
from tllm.workgraph.frontend import (
    CaptureStatus,
    LocalRef,
    SinkRef,
    StateScope,
    extern,
    flow,
    hidden,
    local,
    request_prompt_idx,
    result,
    sink,
)


def _program_instance() -> torch.nn.Module:
    class Mini(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        @flow(
            inputs={"source": hidden(layer=0, site="block_output", phase="decode")},
            outputs={"prediction": sink("guidance"), "loss": local()},
            state="distiller",
        )
        def forward(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.proj(source)
            return out, out.square().mean()

        @extern(
            inputs={"loss": result("forward", "loss")},
            after=("forward",),
            state="distiller",
            deadline=TimingDeadline.NEXT_PREPARE,
        )
        def backward(self, loss: torch.Tensor) -> None:
            return None

    return Mini()


def _declare(instance: torch.nn.Module | None = None) -> ProgramDecl:
    return declare_program(
        instance if instance is not None else _program_instance(),
        states={
            "distiller": StateSpec(
                scope=StateScope.REQUEST,
                key=request_prompt_idx(),
                optimizer=torch.optim.SGD,
            )
        },
    )


class ProgramValueTest(unittest.TestCase):
    def test_flow_outputs_become_named_values(self) -> None:
        decl = _declare()
        by_name = {value.name: value for value in decl.values}
        self.assertEqual(tuple(by_name), ("prediction", "loss"))
        self.assertIsInstance(by_name["prediction"].ref, SinkRef)
        self.assertIsInstance(by_name["loss"].ref, LocalRef)
        self.assertEqual(by_name["prediction"].producer, "forward")
        self.assertEqual(by_name["loss"].consumers, ("backward",))
        self.assertEqual(by_name["prediction"].consumers, ())

    def test_states_carry_bound_nodes(self) -> None:
        decl = _declare()
        self.assertEqual(tuple(state.name for state in decl.states), ("distiller",))
        self.assertEqual(decl.states[0].nodes, ("forward", "backward"))
        self.assertIs(decl.states[0].spec.optimizer, torch.optim.SGD)


class ProgramEdgeTest(unittest.TestCase):
    def test_data_control_state_edges(self) -> None:
        decl = _declare()
        data = [edge for edge in decl.edges if edge.kind is EdgeKind.DATA]
        control = [edge for edge in decl.edges if edge.kind is EdgeKind.CONTROL]
        state = [edge for edge in decl.edges if edge.kind is EdgeKind.STATE]
        self.assertEqual(
            [(e.source, e.target, e.detail) for e in data],
            [("forward", "backward", "loss->loss")],
        )
        self.assertEqual(
            [(e.source, e.target) for e in control],
            [("forward", "backward")],
        )
        self.assertEqual(
            [(e.source, e.target) for e in state],
            [("forward", "distiller"), ("backward", "distiller")],
        )

    def test_schedule_is_topological_and_deterministic(self) -> None:
        first = _declare()
        second = _declare()
        self.assertEqual(first.schedule, ("forward", "backward"))
        self.assertEqual(first.schedule, second.schedule)
        self.assertEqual(
            [(n.name, n.ordinal) for n in first.nodes],
            [(n.name, n.ordinal) for n in second.nodes],
        )

    def test_ir_holds_no_runtime_objects(self) -> None:
        decl = _declare()
        # Static IR is plain data: no tensors, streams, or CUDA objects.
        for node in decl.nodes:
            self.assertIn(node.kind, ("flow", "extern"))
        for edge in decl.edges:
            self.assertIsInstance(edge.kind, EdgeKind)
            self.assertIsInstance(edge.source, str)
            self.assertIsInstance(edge.target, str)


class ProgramDiagnosticTest(unittest.TestCase):
    def test_unconsumed_local_value_is_pg300(self) -> None:
        decl = _declare()
        # "prediction" escapes to a sink; nothing consumes it inside the
        # program, but SinkRef outputs are not diagnosed. Add an unconsumed
        # local to observe PG300.
        class Extra(torch.nn.Module):
            @flow(
                inputs={"source": hidden(layer=0, site="block_output", phase="decode")},
                outputs={"scratch": local()},
            )
            def forward(self, source: torch.Tensor) -> torch.Tensor:
                return source

        extra = declare_program(Extra(), states={})
        codes = tuple(d.code for d in extra.diagnostics)
        self.assertIn("PG300", codes)
        pg300 = [d for d in extra.diagnostics if d.code == "PG300"][0]
        self.assertEqual(pg300.node, "forward")

    def test_unused_state_is_pg301(self) -> None:
        decl = declare_program(
            _program_instance(),
            states={
                "distiller": StateSpec(scope=StateScope.REQUEST, key=request_prompt_idx()),
                "spare": StateSpec(scope=StateScope.REPLICA),
            },
        )
        pg301 = [d for d in decl.diagnostics if d.code == "PG301"]
        self.assertEqual(len(pg301), 1)
        self.assertIn("spare", pg301[0].message)

    def test_valid_program_has_no_diagnostics(self) -> None:
        self.assertEqual(_declare().diagnostics, ())


class TransparencyRegressionTest(unittest.TestCase):
    def test_program_methods_call_through_as_plain_python(self) -> None:
        instance = _program_instance()
        source = torch.ones(2, 4)
        prediction, loss = instance.forward(source)
        self.assertEqual(tuple(prediction.shape), (2, 4))
        self.assertEqual(loss.ndim, 0)
        self.assertIsNone(instance.backward(loss))

    def test_bound_method_discovery(self) -> None:
        instance = _program_instance()
        decl = _declare(instance)
        forward_node = decl.nodes[0]
        self.assertIs(forward_node.function.__self__, instance)

    def test_legacy_capture_surface_untouched(self) -> None:
        @flow
        def bare(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        self.assertTrue(callable(bare.capture))
        self.assertIsNone(getattr(bare, "flow_program", None))
        captured = bare.capture(torch.ones(2))
        self.assertEqual(captured.status, CaptureStatus.CAPTURED)
        self.assertEqual(captured.diagnostics, ())


if __name__ == "__main__":
    unittest.main()

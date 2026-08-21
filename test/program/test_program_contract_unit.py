#!/usr/bin/env python3
"""Contract tests for the Program declaration surface (plan Phase 0).

Covers the extended flow/extern decorators, the declaration vocabulary
(hidden/sink/local/result/request_prompt_idx), and declare_program static
validation: unknown references (PG100), dependency cycles (PG110),
duplicate producers (PG120), unsupported optimizers (PG160), and the
no-silent-fallback rule (compile failures raise; nothing half-registers).
"""

from __future__ import annotations

import unittest

import torch

from tllm.ports.base import TimingDeadline
from tllm.workgraph.frontend import (
    Capability,
    Device,
    Effect,
    StateScope,
    extern,
    flow,
    hidden,
    local,
    request_prompt_idx,
    result,
    sink,
)
from tllm.workgraph.frontend import HiddenSource, LocalRef, PromptIdxKey, ResultRef, SinkRef
from tllm.workgraph.lowering import WorkGraphError


class DeclarationVocabularyTest(unittest.TestCase):
    def test_hidden_validates_arguments(self) -> None:
        src = hidden(layer=0, site="block_output", phase="decode")
        self.assertIsInstance(src, HiddenSource)
        with self.assertRaises(TypeError):
            hidden(layer="0", site="block_output", phase="decode")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            hidden(layer=0, site="", phase="decode")
        with self.assertRaises(ValueError):
            hidden(layer=0, site="block_output", phase="")

    def test_sink_and_local(self) -> None:
        self.assertIsInstance(sink("guidance"), SinkRef)
        self.assertIsInstance(local(), LocalRef)
        with self.assertRaises(ValueError):
            sink("")
        with self.assertRaises(TypeError):
            sink(123)  # type: ignore[arg-type]

    def test_result_ref(self) -> None:
        ref = result("forward", "loss")
        self.assertIsInstance(ref, ResultRef)
        with self.assertRaises(ValueError):
            result("", "loss")
        with self.assertRaises(ValueError):
            result("forward", "")

    def test_request_prompt_idx_key(self) -> None:
        self.assertIsInstance(request_prompt_idx(), PromptIdxKey)


class FlowDeclarationTest(unittest.TestCase):
    def test_flow_records_program_metadata(self) -> None:
        @flow(
            inputs={
                "source": hidden(layer=0, site="block_output", phase="decode"),
                "target": hidden(layer=-1, site="block_output", phase="decode"),
            },
            outputs={"prediction": sink("guidance"), "loss": local()},
            state="distiller",
        )
        def forward(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return source, target.sum(dim=-1)

        decl = getattr(forward, "flow_program")
        self.assertEqual(tuple(name for name, _ in decl.inputs), ("source", "target"))
        self.assertEqual(tuple(name for name, _ in decl.outputs), ("prediction", "loss"))
        self.assertEqual(decl.state, "distiller")
        # call transparency: a decorated function still runs plain Python
        out = forward(torch.ones(2, 3), torch.zeros(2, 3))
        self.assertEqual(int(out[1].shape[0]), 2)

    def test_flow_input_keys_must_match_parameter_names(self) -> None:
        with self.assertRaises(ValueError):
            @flow(inputs={"bogus": hidden(layer=0, site="block_output", phase="decode")})
            def forward(source: torch.Tensor) -> torch.Tensor:
                return source

    def test_flow_input_values_must_be_hidden_sources(self) -> None:
        with self.assertRaises(TypeError):
            @flow(inputs={"source": sink("guidance")})
            def forward(source: torch.Tensor) -> torch.Tensor:
                return source

    def test_flow_output_values_must_be_sink_or_local(self) -> None:
        with self.assertRaises(TypeError):
            @flow(outputs={"prediction": hidden(layer=0, site="block_output", phase="decode")})
            def forward(source: torch.Tensor) -> torch.Tensor:
                return source

    def test_flow_state_must_be_a_name(self) -> None:
        with self.assertRaises(TypeError):
            @flow(state=42)  # type: ignore[arg-type]
            def forward(source: torch.Tensor) -> torch.Tensor:
                return source

    def test_legacy_flow_surface_unchanged(self) -> None:
        @flow
        def bare(x: torch.Tensor) -> torch.Tensor:
            return x

        self.assertIsNone(getattr(bare, "flow_program", None))
        self.assertEqual(int(bare(torch.ones(1)).sum()), 1)

        @flow()
        def kwargs_only(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        self.assertIsNone(getattr(kwargs_only, "flow_program", None))


class ExternDeclarationTest(unittest.TestCase):
    def test_extern_records_program_metadata_with_defaults(self) -> None:
        @extern(
            inputs={"loss": result("forward", "loss")},
            after=("forward",),
            state="distiller",
            deadline=TimingDeadline.NEXT_PREPARE,
        )
        def backward(loss: torch.Tensor) -> None:
            return None

        decl = getattr(backward, "extern_program")
        self.assertEqual(tuple(name for name, _ in decl.inputs), ("loss",))
        self.assertEqual(decl.after, ("forward",))
        self.assertEqual(decl.state, "distiller")
        self.assertIs(decl.deadline, TimingDeadline.NEXT_PREPARE)
        # Program-mode defaults (plan: WRITE_STATE / CUDA / capture-eligible)
        contract = backward.extern_contract
        self.assertIs(contract.effect, Effect.WRITE_STATE)
        self.assertIs(contract.device, Device.CUDA)
        self.assertIs(contract.capability, Capability.CAPTURE_SAFE)

    def test_extern_after_must_be_flow_names(self) -> None:
        with self.assertRaises(TypeError):
            @extern(after=(42,), state="distiller")  # type: ignore[arg-type]
            def backward(loss: torch.Tensor) -> None:
                return None

    def test_extern_deadline_must_be_timing_deadline(self) -> None:
        with self.assertRaises(TypeError):
            @extern(after=("forward",), state="distiller", deadline="soon")  # type: ignore[arg-type]
            def backward(loss: torch.Tensor) -> None:
                return None

    def test_extern_input_values_must_be_result_or_hidden(self) -> None:
        with self.assertRaises(TypeError):
            @extern(inputs={"loss": local()}, state="distiller")
            def backward(loss: torch.Tensor) -> None:
                return None

    def test_extern_input_keys_must_match_parameter_names(self) -> None:
        with self.assertRaises(ValueError):
            @extern(inputs={"bogus": result("forward", "loss")}, state="distiller")
            def backward(loss: torch.Tensor) -> None:
                return None

    def test_legacy_extern_surface_unchanged(self) -> None:
        @extern(effect=Effect.PURE, state=StateScope.NONE, device=Device.CPU, capability=Capability.EAGER_ONLY)
        def legacy(x: torch.Tensor) -> torch.Tensor:
            return x

        self.assertIsNone(getattr(legacy, "extern_program", None))
        self.assertIs(legacy.extern_contract.effect, Effect.PURE)
        with self.assertRaises(TypeError):
            # legacy mode keeps effect/state/device/capability required
            @extern()  # type: ignore[call-arg]
            def missing(x: torch.Tensor) -> torch.Tensor:
                return x


def _mini_program():
    class Mini(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        @flow(
            inputs={"source": hidden(layer=0, site="block_output", phase="decode")},
            outputs={"loss": local()},
            state="distiller",
        )
        def forward(self, source: torch.Tensor) -> torch.Tensor:
            return self.proj(source).square().mean()

        @extern(
            inputs={"loss": result("forward", "loss")},
            after=("forward",),
            state="distiller",
            deadline=TimingDeadline.NEXT_PREPARE,
        )
        def backward(self, loss: torch.Tensor) -> None:
            return None

    return Mini()


class DeclareProgramValidationTest(unittest.TestCase):
    def test_valid_program_declares(self) -> None:
        from tllm.program import StateSpec, declare_program

        decl = declare_program(
            _mini_program(),
            states={
                "distiller": StateSpec(
                    scope=StateScope.REQUEST,
                    key=request_prompt_idx(),
                    optimizer=torch.optim.Adam,
                )
            },
        )
        self.assertEqual(tuple(node.name for node in decl.nodes), ("forward", "backward"))
        self.assertEqual(tuple(node.ordinal for node in decl.nodes), (0, 1))

    def test_unknown_result_reference_fails_pg100(self) -> None:
        from tllm.program import StateSpec, declare_program

        class Bad(torch.nn.Module):
            @extern(
                inputs={"loss": result("nope", "loss")},
                state="distiller",
                deadline=TimingDeadline.NEXT_PREPARE,
            )
            def backward(self, loss: torch.Tensor) -> None:
                return None

        with self.assertRaises(WorkGraphError) as ctx:
            declare_program(Bad(), states={"distiller": StateSpec(scope=StateScope.REQUEST, key=request_prompt_idx())})
        self.assertEqual(ctx.exception.code, "PG100")

    def test_unknown_state_name_fails_pg100(self) -> None:
        from tllm.program import declare_program

        with self.assertRaises(WorkGraphError) as ctx:
            declare_program(_mini_program(), states={})
        self.assertEqual(ctx.exception.code, "PG100")

    def test_dependency_cycle_fails_pg110(self) -> None:
        from tllm.program import declare_program

        class Cyclic(torch.nn.Module):
            @extern(inputs={"x": result("step_b", "out")}, state="distiller")
            def step_a(self, x: torch.Tensor) -> None:
                return None

            @extern(
                inputs={"y": result("step_a", "out")},
                after=("step_a",),
                state="distiller",
            )
            def step_b(self, y: torch.Tensor) -> None:
                return None

        from tllm.program import StateSpec

        with self.assertRaises(WorkGraphError) as ctx:
            declare_program(
                Cyclic(),
                states={"distiller": StateSpec(scope=StateScope.REQUEST, key=request_prompt_idx())},
            )
        self.assertEqual(ctx.exception.code, "PG110")

    def test_duplicate_producer_fails_pg120(self) -> None:
        from tllm.program import StateSpec, declare_program

        class Dup(torch.nn.Module):
            @flow(
                inputs={"source": hidden(layer=0, site="block_output", phase="decode")},
                outputs={"prediction": sink("guidance")},
            )
            def forward_a(self, source: torch.Tensor) -> torch.Tensor:
                return source

            @flow(
                inputs={"source": hidden(layer=0, site="block_output", phase="decode")},
                outputs={"prediction": sink("guidance")},
            )
            def forward_b(self, source: torch.Tensor) -> torch.Tensor:
                return source

        with self.assertRaises(WorkGraphError) as ctx:
            declare_program(Dup(), states={})
        self.assertEqual(ctx.exception.code, "PG120")

    def test_unsupported_optimizer_fails_pg160(self) -> None:
        from tllm.program import StateSpec, declare_program

        with self.assertRaises(WorkGraphError) as ctx:
            declare_program(
                _mini_program(),
                states={
                    "distiller": StateSpec(
                        scope=StateScope.REQUEST,
                        key=request_prompt_idx(),
                        optimizer=torch.optim.RMSprop,
                    )
                },
            )
        self.assertEqual(ctx.exception.code, "PG160")

    def test_no_silent_fallback(self) -> None:
        from tllm.program import StateSpec, declare_program

        class Bad(torch.nn.Module):
            @extern(inputs={"loss": result("nope", "loss")}, state="distiller")
            def backward(self, loss: torch.Tensor) -> None:
                return None

        # compile failure raises; nothing partial is returned or registered
        with self.assertRaises(WorkGraphError):
            declare_program(Bad(), states={"distiller": StateSpec(scope=StateScope.REQUEST, key=request_prompt_idx())})


if __name__ == "__main__":
    unittest.main()

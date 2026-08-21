from __future__ import annotations

import dataclasses
import inspect
import unittest
from enum import Enum
from typing import Any

import torch

from tllm.workgraph.frontend import (
    ArtifactKey,
    BeforeNextUse,
    Capability,
    CapturedFlow,
    Device,
    Effect,
    FlowCaptureCode,
    FlowCaptureDiagnostic,
    PortBindingDesc,
    PortBindingDirection,
    ShapeGuard,
    SourceSpan,
    StateScope,
    extern,
    flow,
)
from tllm.ports.port_id import PortId
from tllm.workgraph.signature import structural_signature


class WorkGraphFrontendContractTest(unittest.TestCase):
    def test_flow_call_is_transparent_and_capture_is_control_plane_only(self) -> None:
        calls: list[int] = []

        def body(value: int) -> int:
            calls.append(value)
            return value + 1

        wrapped = flow(body)
        self.assertEqual(wrapped(4), 5)
        self.assertEqual(calls, [4])
        self.assertNotIsInstance(wrapped(5), CapturedFlow)
        self.assertEqual(calls, [4, 5])
        self.assertTrue(callable(wrapped.capture))
        with self.assertRaises(TypeError):
            wrapped.capture()

    def test_capture_returns_typed_shadow_artifact(self) -> None:
        @flow
        def add_one(value: torch.Tensor) -> torch.Tensor:
            return value + 1

        captured = add_one.capture(torch.ones(2))
        self.assertIsInstance(captured, CapturedFlow)
        self.assertIsInstance(captured.artifact_key, ArtifactKey)
        self.assertIsInstance(captured.source_span, SourceSpan)
        self.assertIsInstance(captured.shape_guard, ShapeGuard)
        self.assertIsInstance(captured.inputs, tuple)
        self.assertIsInstance(captured.outputs, tuple)
        self.assertTrue(all(isinstance(binding, PortBindingDesc) for binding in captured.inputs))
        self.assertTrue(all(isinstance(binding, PortBindingDesc) for binding in captured.outputs))
        self.assertIsInstance(captured.diagnostics, tuple)
        self.assertTrue(
            all(isinstance(diagnostic, FlowCaptureDiagnostic) for diagnostic in captured.diagnostics)
        )
        self.assertEqual(captured.node_kind.value, "compute")
        self.assertEqual(captured.mobility.value, "pinned")
        self.assertEqual(captured.graph_safety.value, "capture_safe")
        self.assertEqual(captured.effects, (Effect.PURE,))
        self.assertNotIn("0x", str(captured.artifact_key))

    def test_public_frontend_descriptors_are_frozen_and_enum_contracts_are_typed(self) -> None:
        descriptor_types = (
            CapturedFlow,
            FlowCaptureDiagnostic,
            SourceSpan,
            ShapeGuard,
            ArtifactKey,
            PortBindingDesc,
        )
        for descriptor_type in descriptor_types:
            with self.subTest(descriptor=descriptor_type.__name__):
                self.assertTrue(dataclasses.is_dataclass(descriptor_type))
                params = getattr(descriptor_type, "__dataclass_params__")
                self.assertTrue(params.frozen)
                self.assertTrue(hasattr(descriptor_type, "__slots__"))

        self.assertTrue(dataclasses.is_dataclass(BeforeNextUse))
        self.assertTrue(type(BeforeNextUse).__dataclass_params__.frozen)
        for enum_type in (FlowCaptureCode, Effect, StateScope, Device, Capability):
            with self.subTest(enum=enum_type.__name__):
                self.assertTrue(issubclass(enum_type, Enum))
                self.assertTrue(tuple(enum_type))

    def test_extern_uses_annotations_and_typed_effect_state_capability(self) -> None:
        effect = next(iter(Effect))
        state = next(iter(StateScope))
        device = next(iter(Device))
        capability = next(iter(Capability))

        @extern(effect=effect, state=state, device=device, capability=capability)
        def typed_extern(value: int) -> int:
            return value

        self.assertEqual(typed_extern(7), 7)
        signature = inspect.signature(typed_extern)
        self.assertIs(signature.parameters["value"].annotation, int)
        self.assertIs(signature.return_annotation, int)
        contract: Any = getattr(typed_extern, "extern_contract")
        self.assertIs(contract.effect, effect)
        self.assertIs(contract.state, state)
        self.assertIs(contract.device, device)
        self.assertIs(contract.capability, capability)
        self.assertIsInstance(contract.effect, Effect)
        self.assertIsInstance(contract.state, StateScope)
        self.assertIsInstance(contract.capability, Capability)

    def test_extern_rejects_untyped_or_string_policy_contracts(self) -> None:
        effect = next(iter(Effect))
        state = next(iter(StateScope))
        device = next(iter(Device))
        capability = next(iter(Capability))

        with self.assertRaises((TypeError, ValueError)):
            extern(effect=effect, state=state, device=device, capability=capability)(
                lambda value: value
            )

        with self.assertRaises((TypeError, ValueError)):
            extern(effect="pure", state="local", device="cpu", capability="eager_only")(  # type: ignore[arg-type]
                lambda value: value
            )

    def test_extern_rejects_input_binding_to_an_unknown_parameter_position(self) -> None:
        binding = PortBindingDesc(
            port=PortId("extern", "input"),
            direction=PortBindingDirection.INPUT,
            value_path=(99,),
        )

        with self.assertRaises(ValueError):

            @extern(
                effect=Effect.WRITE_STATE,
                state=StateScope.REPLICA,
                device=Device.CPU,
                capability=Capability.EAGER_ONLY,
                bindings=(binding,),
            )
            def invalid_binding(value: int) -> int:
                return value

    def test_port_bindings_are_structured_unique_and_signature_stable(self) -> None:
        binding = PortBindingDesc(
            port=PortId("capture", "input"),
            direction=PortBindingDirection.INPUT,
            value_path=(0,),
        )

        @flow(bindings=(binding,))
        def identity(value: torch.Tensor) -> torch.Tensor:
            return value

        first = identity.capture(torch.ones(1))
        second = identity.capture(torch.zeros(1))
        self.assertEqual(first.inputs, (binding,))
        self.assertEqual(structural_signature(first), structural_signature(second))

        with self.assertRaises(TypeError):
            PortBindingDesc(  # type: ignore[arg-type]
                port="capture:input@1",
                direction=PortBindingDirection.INPUT,
                value_path=(0,),
            )

        with self.assertRaises(ValueError):
            dataclasses.replace(first, inputs=(binding, binding))


if __name__ == "__main__":
    unittest.main()

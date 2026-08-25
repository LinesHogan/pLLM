#!/usr/bin/env python3
"""Backend contract tests for ModuleWeightView wrapper overlays."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from tllm.backends.protocol import (
    GraphSafety,
    ModuleWeightOverlayCapabilities,
    RunnerBinding,
)
from tllm.backends.vllm.v027_mrv1 import V027MRv1Driver
from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
from tllm.ports.module_weight_view import ModuleWeightViewLocator
from tllm.ports.base import ConsumerFlow
from tllm.ports.module_weight_view import ModuleWeightView
from tllm.runtime.dispatch_plan import DispatchPlan


class _RestorableHandle:
    def __init__(self, module: object, original_forward: object) -> None:
        self.module = module
        self.original_forward = original_forward
        self.restore_count = 0

    def restore(self) -> None:
        setattr(self.module, "forward", self.original_forward)
        self.restore_count += 1


def _runner(layer_count: int = 2) -> SimpleNamespace:
    layers = []
    for layer_idx in range(layer_count):
        mlp = SimpleNamespace()

        def _forward(value: object, *, _layer_idx: int = layer_idx) -> tuple[int, object]:
            return _layer_idx, value

        mlp.forward = _forward
        layers.append(SimpleNamespace(mlp=mlp))
    return SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=layers)))


class ModuleWeightOverlayBackendUnitTest(unittest.TestCase):
    def test_same_locator_same_order_is_rejected_by_rust_placement(self) -> None:
        class _Writer:
            def __init__(self, consumer_id: str) -> None:
                self.consumer_id = consumer_id

            def flows(self) -> tuple[ConsumerFlow, ...]:
                return (
                    ConsumerFlow(
                        reads=(),
                        writes=(
                            ModuleWeightView.write(
                                layer=0,
                                module="mlp",
                                phase="prefill",
                                order=0,
                            ),
                        ),
                        window="same_step",
                    ),
                )

        with self.assertRaisesRegex(RuntimeError, "duplicate order 0"):
            DispatchPlan.build([_Writer("first"), _Writer("second")])  # type: ignore[list-item]

    def test_v027_drivers_declare_typed_eager_tp1_overlay_capability(self) -> None:
        for driver in (V027MRv1Driver(), V027MRv2Driver()):
            overlay = driver.capabilities().module_weight_overlay
            self.assertEqual(
                overlay,
                ModuleWeightOverlayCapabilities(
                    modules=frozenset({"mlp"}),
                    write_modes=frozenset({"wrapper"}),
                    phases=frozenset({"prefill", "decode"}),
                    request_keyed=True,
                    graph_safety=GraphSafety.EAGER_ONLY,
                    tensor_parallel_sizes=frozenset({1}),
                ),
            )
            assert overlay is not None
            self.assertIn("module_weight_view/wrapper/prefill", overlay.capability_names())
            self.assertIn("module_weight_view/wrapper/decode", overlay.capability_names())

    def test_negative_layer_resolves_and_repeated_install_is_idempotent(self) -> None:
        driver = V027MRv1Driver()
        runner = _runner()
        binding = RunnerBinding(driver=driver, runner=runner, capabilities=driver.capabilities())
        locator = ModuleWeightViewLocator(layer=-1, module="mlp", phase="prefill")
        target = driver.resolve_module_weight_view(binding=binding, locator=locator)
        expected = runner.model.model.layers[1].mlp
        self.assertIs(target.module, expected)
        self.assertEqual(target.layer, 1)
        self.assertEqual(target.path, "model.layers.1.mlp")
        calls = 0

        def _install(module: object) -> _RestorableHandle:
            nonlocal calls
            calls += 1
            original = getattr(module, "forward")
            setattr(module, "forward", lambda value: ("wrapped", value))
            return _RestorableHandle(module, original)

        first = driver.install_module_weight_overlay(binding=binding, locator=locator, installer=_install)
        second = driver.install_module_weight_overlay(binding=binding, locator=locator, installer=_install)

        self.assertIs(first, second)
        self.assertEqual(calls, 1)
        self.assertEqual(expected.forward("x"), ("wrapped", "x"))

    def test_unbind_restores_original_forward_idempotently(self) -> None:
        driver = V027MRv2Driver()
        runner = _runner(1)
        binding = RunnerBinding(driver=driver, runner=runner, capabilities=driver.capabilities())
        locator = ModuleWeightViewLocator(layer=0, module="mlp", phase="decode")
        module = runner.model.model.layers[0].mlp
        original = module.forward

        def _install(target: object) -> _RestorableHandle:
            setattr(target, "forward", lambda value: ("wrapped", value))
            return _RestorableHandle(target, original)

        handle = driver.install_module_weight_overlay(binding=binding, locator=locator, installer=_install)
        driver.unbind(binding)
        driver.unbind(binding)

        self.assertFalse(binding.active)
        self.assertIs(module.forward, original)
        self.assertEqual(handle.restore_count, 1)

    def test_install_failure_rolls_back_forward(self) -> None:
        driver = V027MRv1Driver()
        runner = _runner(1)
        binding = RunnerBinding(driver=driver, runner=runner, capabilities=driver.capabilities())
        locator = ModuleWeightViewLocator(layer=0, module="mlp", phase="prefill")
        module = runner.model.model.layers[0].mlp
        original = module.forward

        def _broken_install(target: object) -> object:
            setattr(target, "forward", lambda value: ("broken", value))
            raise RuntimeError("installer failed")

        with self.assertRaisesRegex(RuntimeError, "installer failed"):
            driver.install_module_weight_overlay(
                binding=binding,
                locator=locator,
                installer=_broken_install,
            )

        self.assertIs(module.forward, original)
        self.assertEqual(binding.module_weight_overlay_handles, {})


if __name__ == "__main__":
    unittest.main()

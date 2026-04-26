#!/usr/bin/env python3
"""Unit tests for In-Place TTT runtime ports."""

from __future__ import annotations

import unittest


class InPlaceTTTPortsUnitTest(unittest.TestCase):
    def test_inplace_ttt_ports_expose_expected_kinds(self) -> None:
        from tllm.ports.base import PortKind
        from tllm.ports.hidden_target import HiddenTarget
        from tllm.ports.mlp_activation import MLPActivation
        from tllm.ports.module_weight_view import ModuleWeightView

        self.assertEqual(MLPActivation.read(layer=0).kind, PortKind.MLP_ACTIVATION)
        self.assertEqual(HiddenTarget.read(layer=0).kind, PortKind.HIDDEN_TARGET)
        self.assertEqual(ModuleWeightView.read(layer=0, module="down_proj").kind, PortKind.MODULE_WEIGHT_VIEW)

    def test_inplace_ttt_locators_reject_unsupported_values(self) -> None:
        from tllm.ports.hidden_target import HiddenTarget
        from tllm.ports.mlp_activation import MLPActivation
        from tllm.ports.module_weight_view import ModuleWeightView

        with self.assertRaisesRegex(ValueError, "unsupported mlp activation phase"):
            MLPActivation.read(layer=0, phase="train")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "unsupported hidden target phase"):
            HiddenTarget.read(layer=0, phase="train")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "module must be non-empty"):
            ModuleWeightView.read(layer=0, module="")


if __name__ == "__main__":
    unittest.main()

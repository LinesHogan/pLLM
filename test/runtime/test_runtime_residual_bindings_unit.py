#!/usr/bin/env python3
"""Unit tests for residual locator/path bindings in the runtime layer."""

from __future__ import annotations

import unittest

from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.ports import residual_bindings


class RuntimeResidualBindingsUnitTest(unittest.TestCase):
    def test_build_resolved_bindings_maps_logical_locators_to_runtime_paths(self) -> None:
        cfg = type(
            "Config",
            (),
            {
                "source_layer_path": "model.model.layers[0].input_layernorm",
                "target_layer_path": "model.model.layers[-1].input_layernorm",
            },
        )()
        bindings = residual_bindings.build_resolved_bindings(
            raw_paths_by_locator=residual_bindings.default_raw_paths_from_config(cfg),
            resolve_alias={
                "model.model.layers[0].input_layernorm": "layers.0",
                "model.model.layers[-1].input_layernorm": "layers.1",
            },
            required=set(),
        )

        self.assertEqual(list(bindings.keys()), ["layers.0", "layers.1"])
        self.assertEqual(bindings["layers.0"].locator, ResidualLocator(layer=0, site="block_output", phase="decode"))
        self.assertTrue(bindings["layers.0"].include_request_meta)
        self.assertEqual(bindings["layers.1"].locator, ResidualLocator(layer=-1, site="block_output", phase="decode"))
        self.assertFalse(bindings["layers.1"].include_request_meta)

    def test_unsupported_logical_locator_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsupported logical residual locator"):
            residual_bindings.build_raw_tap_paths(
                raw_paths_by_locator={},
                required={(0, "post", "decode")},
            )


if __name__ == "__main__":
    unittest.main()

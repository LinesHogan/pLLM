#!/usr/bin/env python3
"""Unit tests for generic residual runtime setup extraction."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime.ports import residual_runtime_setup


class RuntimeResidualRuntimeSetupUnitTest(unittest.TestCase):
    def test_native_setup_uses_only_explicit_resource_locators(self) -> None:
        locator = ResidualLocator(
            layer=0,
            site="block_output",
            phase="decode",
        )
        raw_path = "model.layers.0"
        layer = torch.nn.Linear(4, 4)

        class Model(torch.nn.Module):
            pass

        model = Model()
        model.mapping = {raw_path: layer}
        model.config = type("Cfg", (), {"hidden_size": 4})()
        runtime = type("Runtime", (), {})()
        runtime.config = type(
            "Config",
            (),
            {
                "tap_layer_paths": [],
                "graph_scratch_rows": 2,
                "compact_capture_lane": False,
                "enable_distiller_intervention": False,
            },
        )()
        runtime.residual_raw_paths = {locator: raw_path}
        runtime.dispatch_plan = None
        runtime.native_schedule_graph = type("Graph", (), {"tasks": ()})()
        core = type(
            "Core",
            (),
            {
                "RUNTIME": runtime,
                "_resolve_module_by_path_with_fallback": staticmethod(
                    lambda model, path: (model.mapping[path], path)
                ),
                "_infer_hidden_dtype": staticmethod(lambda _layer: torch.float32),
                "_runner_uses_compilation_or_cudagraph": staticmethod(
                    lambda _runner: False
                ),
            },
        )()
        runner = type(
            "Runner",
            (),
            {
                "model": model,
                "device": torch.device("cpu"),
                "max_num_reqs": 2,
                "get_model": lambda self: model,
            },
        )()

        setup = residual_runtime_setup.resolve_runtime_setup(core=core, runner=runner)

        self.assertEqual(tuple(setup.runtime_bindings), (raw_path,))
        self.assertEqual(setup.runtime_bindings[raw_path].locator, locator)

    def test_apply_runtime_setup_populates_runtime_state_and_installs_hooks(self) -> None:
        runtime = type("Runtime", (), {})()
        runtime.config = type(
            "Config",
            (),
            {
                "tap_layer_paths": [],
                "source_layer_path": "model.model.layers[0].input_layernorm",
                "target_layer_path": "model.model.layers[-1].input_layernorm",
                "graph_scratch_rows": 8,
                "compact_capture_lane": False,
                "enable_distiller_intervention": False,
            },
        )()
        runtime.residual_raw_paths = {
            ResidualLocator(layer=0, site="block_output", phase="decode"): "model.model.layers[0].input_layernorm",
            ResidualLocator(layer=-1, site="block_output", phase="decode"): "model.model.layers[-1].input_layernorm",
        }
        runtime.dispatch_plan = None
        runtime.tap_layers = {}
        runtime.tap_scratch = {}
        runtime.tap_decode_hidden = {}
        runtime.residual_bindings = {}
        runtime.source_resolved_path = ""
        runtime.target_resolved_path = ""
        runtime.launch_consumer_from_hooks = True
        runtime.native_schedule_graph = None
        runtime.native_hidden_path_aliases = {}
        runtime.decode_row_idx = None
        runtime.decode_valid_mask = None
        core = type(
            "Core",
            (),
            {
                "RUNTIME": runtime,
                "_resolve_module_by_path_with_fallback": staticmethod(lambda model, raw_path: (model.mapping[raw_path], model.alias[raw_path])),
                "_infer_hidden_dtype": staticmethod(lambda layer: torch.float32),
                "_runner_uses_compilation_or_cudagraph": staticmethod(lambda runner: False),
            },
        )()
        layer0 = torch.nn.Linear(4, 4)
        layer1 = torch.nn.Linear(4, 4)

        class Model(torch.nn.Module):
            pass

        model = Model()
        model.mapping = {
            "model.model.layers[0].input_layernorm": layer0,
            "model.model.layers[-1].input_layernorm": layer1,
        }
        model.alias = {
            "model.model.layers[0].input_layernorm": "layers.0",
            "model.model.layers[-1].input_layernorm": "layers.1",
        }
        model.config = type("Cfg", (), {"hidden_size": 4})()
        runner = type(
            "Runner",
            (),
            {
                "model": model,
                "device": torch.device("cpu"),
                "max_num_reqs": 8,
                "get_model": lambda self: model,
            },
        )()
        setup = residual_runtime_setup.resolve_runtime_setup(core=core, runner=runner)

        with mock.patch.object(residual_runtime_setup._residual_capture_hooks, "install_layer_forward_taps") as p_install:
            residual_runtime_setup.apply_runtime_setup(core=core, runner=runner, setup=setup)

        self.assertEqual(core.RUNTIME.source_resolved_path, "layers.0")
        self.assertEqual(core.RUNTIME.target_resolved_path, "layers.1")
        self.assertIn("layers.0", core.RUNTIME.residual_bindings)
        self.assertEqual(tuple(core.RUNTIME.decode_row_idx.shape), (8,))
        self.assertEqual(tuple(core.RUNTIME.decode_valid_mask.shape), (8, 1))
        p_install.assert_called_once()


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit tests for residual capture forward-hook installation."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.runtime.ports import residual_capture_hooks


class _DecodeExecutor:
    capture_resident_producer_ids: tuple[str, ...] = ()

    @staticmethod
    def run_layer_post(ctx) -> None:
        residual_capture_hooks.materialize_decode_residual(
            runtime=ctx.runtime,
            runner=ctx.runner,
            layer_path=ctx.layer_path,
            hidden=ctx.hidden,
        )


class RuntimeResidualCaptureHooksUnitTest(unittest.TestCase):
    def test_capture_resident_tap_uses_module_buffers_and_salts_aot_identity(self) -> None:
        class _Layer(torch.nn.Module):
            def forward(self, x):
                return x + 1

        class _CompiledOwner(torch.nn.Module):
            def __init__(self, layer):
                super().__init__()
                self.layer = layer
                self.compiled = False
                self.aot_compiled_fn = None

            def aot_compile(self):
                raise AssertionError("not called")

            def forward(self, x):
                return self.layer(x)

        layer = _Layer()
        model = _CompiledOwner(layer)
        runtime = type("Runtime", (), {})()
        runtime.write_leases = {}
        runtime.decode_row_idx = torch.tensor([2, 0], dtype=torch.long)
        runtime.decode_valid_mask = torch.ones((2, 1), dtype=torch.float32)
        runtime.tap_decode_hidden = {
            "layers.0": torch.empty((2, 3), dtype=torch.float32)
        }
        runtime.tap_decode_hidden_compact = {}
        runtime.decode_compact_row_idx = None
        runtime.decode_compact_count = 0
        runtime.capture_full_residual_rows = True
        runtime.producer_executor = type(
            "Executor",
            (),
            {"capture_resident_producer_ids": ("tllm.residual.decode",)},
        )()
        runtime.placement_stats = type(
            "Stats", (), {"capture_resident_stage_install_count": 0}
        )()
        core = type("Core", (), {"RUNTIME": runtime})()
        runner = type(
            "Runner", (), {"device": torch.device("cpu"), "model": model}
        )()

        from tllm.runtime.graph_policy import GraphMode

        original_qualname = model.forward.__qualname__
        with mock.patch(
            "tllm.runtime.graph_policy.resolve_graph_mode",
            return_value=GraphMode.FULL,
        ):
            residual_capture_hooks.install_layer_forward_taps(
                core=core,
                runner=runner,
                resolved_layers={"layers.0": layer},
            )

        x = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        self.assertTrue(torch.equal(model(x), x + 1))
        self.assertTrue(
            torch.equal(
                runtime.tap_decode_hidden["layers.0"],
                torch.tensor([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]),
            )
        )
        self.assertIs(getattr(layer, "_tllm_decode_row_idx"), runtime.decode_row_idx)
        self.assertIs(
            getattr(layer, "_tllm_decode_residual"),
            runtime.tap_decode_hidden["layers.0"],
        )
        self.assertNotIn("_tllm_decode_row_idx", layer.state_dict())
        self.assertRegex(
            model.forward.__qualname__,
            rf"^{original_qualname}__tllm_capture_[0-9a-f]{{16}}$",
        )
        self.assertEqual(runtime.placement_stats.capture_resident_stage_install_count, 1)

    def test_install_layer_forward_taps_captures_decode_rows_and_calls_runtime_helpers(self) -> None:
        class _Layer(torch.nn.Module):
            def forward(self, x):
                return x + 1

        layer = _Layer()
        runtime = type("Runtime", (), {})()
        runtime.write_leases = {}
        runtime.decode_row_idx = torch.tensor([2, 0], dtype=torch.long)
        runtime.decode_valid_mask = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
        runtime.decode_count = 2
        runtime.tap_decode_hidden = {"layers.0": torch.empty((2, 4), dtype=torch.float32)}
        runtime.tap_decode_hidden_compact = {}
        runtime.decode_compact_row_idx = None
        runtime.decode_compact_count = 0
        runtime.capture_full_residual_rows = True
        runtime.producer_executor = _DecodeExecutor()
        runtime.launch_consumer_from_hooks = True
        runtime.dispatch_plan = type(
            "Plan",
            (),
            {"has_active_targets": staticmethod(lambda: True)},
        )()
        runtime.consumer = None
        runtime.source_resolved_path = "layers.0"
        runtime.target_resolved_path = "layers.1"
        core = type("Core", (), {"RUNTIME": runtime})()
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(
            residual_capture_hooks._sampler_patch,
            "maybe_capture_source_precompute",
        ) as p_precompute, mock.patch.object(
            residual_capture_hooks._hidden_bridge, "dispatch_layer_lifecycle_events"
        ) as p_events:
            residual_capture_hooks.install_layer_forward_taps(
                core=core,
                runner=runner,
                resolved_layers={"layers.0": layer},
            )
            x = torch.tensor(
                [
                    [10.0, 11.0, 12.0, 13.0],
                    [20.0, 21.0, 22.0, 23.0],
                    [30.0, 31.0, 32.0, 33.0],
                ],
                dtype=torch.float32,
            )
            out = layer(x)

        self.assertTrue(torch.equal(out, x + 1))
        self.assertTrue(
            torch.equal(
                runtime.tap_decode_hidden["layers.0"],
                torch.tensor(
                    [
                        [31.0, 32.0, 33.0, 34.0],
                        [11.0, 12.0, 13.0, 14.0],
                    ],
                    dtype=torch.float32,
                ),
            )
        )
        p_precompute.assert_called_once_with(
            runtime=runtime,
            runner=runner,
            layer_path="layers.0",
        )
        p_events.assert_called_once()

    def test_install_layer_forward_taps_populates_compact_lane_without_mutating_full_lane(self) -> None:
        class _Layer(torch.nn.Module):
            def forward(self, x):
                return x + 1

        layer = _Layer()
        runtime = type("Runtime", (), {})()
        runtime.write_leases = {}
        runtime.decode_row_idx = torch.tensor([2, 0, 1], dtype=torch.long)
        runtime.decode_valid_mask = torch.ones((3, 1), dtype=torch.float32)
        runtime.decode_compact_row_idx = torch.tensor([2, 1], dtype=torch.long)
        runtime.decode_compact_count = 2
        runtime.tap_decode_hidden = {"layers.0": torch.empty((3, 2), dtype=torch.float32)}
        runtime.tap_decode_hidden_compact = {"layers.0": torch.empty((2, 2), dtype=torch.float32)}
        runtime.capture_full_residual_rows = True
        runtime.producer_executor = _DecodeExecutor()
        runtime.launch_consumer_from_hooks = True
        runtime.dispatch_plan = type(
            "Plan",
            (),
            {"has_active_targets": staticmethod(lambda: True)},
        )()
        runtime.consumer = None
        runtime.source_resolved_path = "layers.0"
        runtime.target_resolved_path = "layers.1"
        core = type("Core", (), {"RUNTIME": runtime})()
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        with mock.patch.object(residual_capture_hooks._hidden_bridge, "dispatch_layer_lifecycle_events"):
            residual_capture_hooks.install_layer_forward_taps(
                core=core,
                runner=runner,
                resolved_layers={"layers.0": layer},
            )
            x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
            layer(x)

        self.assertTrue(
            torch.equal(
                runtime.tap_decode_hidden["layers.0"],
                torch.tensor([[6.0, 7.0], [2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                runtime.tap_decode_hidden_compact["layers.0"],
                torch.tensor([[6.0, 7.0], [4.0, 5.0]], dtype=torch.float32),
            )
        )


if __name__ == "__main__":
    unittest.main()

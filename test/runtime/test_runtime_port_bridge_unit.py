#!/usr/bin/env python3
"""Unit tests for bridging current runtime state into port frames."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualLocator, ResidualStream
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.fusion import FusionExecutor
from tllm.runtime.ports.assembler import BundleAssembler
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.scheduling.deferred import DeferredWorkQueue
from tllm.runtime.stats import PlacementStats
from tllm.runtime.torch_adapter import CudaLongHostStaging
from tllm.runtime.vllm_patch import port_runtime_hooks


class RuntimePortBridgeUnitTest(unittest.TestCase):
    def test_native_signed_hidden_requirement_runs_decode_localization(self) -> None:
        localize = mock.Mock()
        runtime = SimpleNamespace(
            producer_executor=None,
            workload_execution=SimpleNamespace(
                requires_decode_hidden=True,
                requires_request_identities=False,
            ),
            dispatch_plan=None,
        )
        core = SimpleNamespace(RUNTIME=runtime)

        port_runtime_hooks.run_prepare_inputs_post_producers(
            core=core,
            runner=SimpleNamespace(),
            localize_decode=localize,
        )

        localize.assert_called_once_with()

    def test_native_request_state_runs_identity_localization_without_legacy_targets(
        self,
    ) -> None:
        localize = mock.Mock()
        runtime = SimpleNamespace(
            producer_executor=None,
            workload_execution=SimpleNamespace(
                requires_decode_hidden=False,
                requires_request_identities=True,
            ),
            dispatch_plan=None,
        )
        core = SimpleNamespace(RUNTIME=runtime)

        port_runtime_hooks.run_prepare_inputs_post_producers(
            core=core,
            runner=SimpleNamespace(),
            localize_decode=localize,
        )

        localize.assert_called_once_with()

    class _StepFlowConsumer(BaseConsumer):
        @property
        def consumer_id(self) -> str:
            return "step_flow_consumer"

        def __init__(self) -> None:
            self.seen = []
            self.feedback_calls = 0

        def flows(self):
            return [
                ConsumerFlow(
                    reads=(
                        ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                        ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                        RequestMeta.read(),
                    ),
                    writes=(),
                    window="out_of_band",
                    bundle_key=("engine_step_id", "phase"),
                )
            ]

        def consume(self, batch, ctx) -> None:
            _ = (batch, ctx)

        def consume_bundle(self, bundle, ctx) -> None:
            self.seen.append(bundle)

        def on_tick(self, event_name, ctx) -> None:
            _ = (event_name, ctx)

        def on_step_end(self, ctx) -> None:
            _ = ctx
            self.feedback_calls += 1

    def _core(self):
        runtime = type("Runtime", (), {})()
        runtime.tap_decode_hidden = {
            "layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]], dtype=torch.float32),
            "layers.1": torch.tensor([[5.0, 6.0], [7.0, 8.0], [88.0, 88.0]], dtype=torch.float32),
        }
        runtime.decode_count = 2
        runtime.decode_prompt_idxs = [10, 11]
        runtime.decode_sample_idxs = [0, 1]
        runtime.decode_request_ids = ["reqA", "reqB"]
        runtime.residual_bindings = {
            "layers.0": ResidualPathBinding(
                locator=ResidualLocator(layer=0, site="block_output", phase="decode"),
                resolved_path="layers.0",
                include_request_meta=True,
            ),
            "layers.1": ResidualPathBinding(
                locator=ResidualLocator(layer=-1, site="block_output", phase="decode"),
                resolved_path="layers.1",
                include_request_meta=False,
            ),
        }
        runtime.source_resolved_path = "layers.0"
        runtime.target_resolved_path = "layers.1"
        runtime.event_step_id = 9
        runtime.dispatch_plan = None
        runtime.deferred_work_queue = DeferredWorkQueue()
        runtime.fusion_executor = FusionExecutor()
        runtime.placement_stats = PlacementStats()
        runtime.path_hotspot_enabled = False
        runtime.workload_execution = None
        return type(
            "Core",
            (),
            {
                "RUNTIME": runtime,
                "record_path_hotspot_cpu": staticmethod(lambda _name, _ms: None),
            },
        )()

    def test_runtime_hook_setup_defers_consumer_resources_until_first_bundle(self) -> None:
        consumer = mock.Mock()
        runtime = SimpleNamespace(
            consumer=consumer,
            launch_consumer_from_hooks=False,
        )
        core = SimpleNamespace(
            RUNTIME=runtime,
            MODEL_HOOK_FLAG="_tllm_hooked",
            MODEL_HOOK_SPEC_ATTR="_tllm_hook_spec",
            _runner_uses_compilation_or_cudagraph=lambda _runner: False,
        )
        class Model(torch.nn.Module):
            def compute_logits(self, *args, **kwargs):
                return args, kwargs

        model = Model()
        runner = SimpleNamespace(model=model, get_model=lambda: model)
        setup = SimpleNamespace(
            resolved_layers=(),
            target_resolved="layers.1",
            hook_spec=("layers.0", "layers.1"),
        )

        with mock.patch.object(
            port_runtime_hooks.active_targets,
            "runtime_has_active_targets",
            return_value=True,
        ), mock.patch.object(
            port_runtime_hooks._residual_runtime_setup,
            "resolve_runtime_setup",
            return_value=setup,
        ), mock.patch.object(
            port_runtime_hooks._residual_runtime_setup,
            "apply_runtime_setup",
        ), mock.patch.object(
            port_runtime_hooks._sampler_patch,
            "bind_runner_sampler",
        ), mock.patch.object(
            port_runtime_hooks._sampler_patch,
            "ensure_sampler_precompute_buffers",
        ):
            port_runtime_hooks.setup_runtime_hooks_if_active(core=core, runner=runner)

        consumer._ensure_runtime_resources.assert_not_called()

    def test_decode_runtime_state_bridges_into_complete_source_target_bundles(self) -> None:
        core = self._core()

        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
        )
        assembler = BundleAssembler(flow)

        source_frames = port_runtime_hooks.build_decode_port_frames(core=core, layer_path="layers.0")
        target_frames = port_runtime_hooks.build_decode_port_frames(core=core, layer_path="layers.1")

        bundles = []
        for frame in source_frames + target_frames:
            bundles.extend(assembler.push(frame))

        self.assertEqual(len(bundles), 2)
        by_request = {bundle.key.request_id: bundle for bundle in bundles}
        self.assertEqual(tuple(by_request["reqA"].entries["source"].tolist()), (1.0, 2.0))
        self.assertEqual(tuple(by_request["reqA"].entries["target"].tolist()), (5.0, 6.0))
        self.assertEqual(by_request["reqA"].entries["request_meta"]["prompt_idx"], 10)
        self.assertEqual(tuple(by_request["reqB"].entries["source"].tolist()), (3.0, 4.0))
        self.assertEqual(tuple(by_request["reqB"].entries["target"].tolist()), (7.0, 8.0))
        self.assertEqual(by_request["reqB"].entries["request_meta"]["sample_idx"], 1)

    def test_runtime_bridge_can_dispatch_step_scope_aggregated_bundle(self) -> None:
        core = self._core()
        consumer = self._StepFlowConsumer()
        core.RUNTIME.dispatch_plan = DispatchPlan.build([consumer])
        runner = type("Runner", (), {"device": torch.device("cpu"), "model": object()})()

        dispatched = port_runtime_hooks.dispatch_decode_port_bundles(core=core, runner=runner)

        self.assertEqual(dispatched, 1)
        self.assertEqual(len(consumer.seen), 1)
        self.assertEqual(consumer.feedback_calls, 1)
        bundle = consumer.seen[0]
        self.assertTrue(torch.equal(bundle.entries["source"], torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(torch.equal(bundle.entries["target"], torch.tensor([[5.0, 6.0], [7.0, 8.0]])))
        self.assertEqual(bundle.entries["request_meta"], [
            {"request_id": "reqA", "prompt_idx": 10, "sample_idx": 0, "phase": "decode", "engine_step_id": 9},
            {"request_id": "reqB", "prompt_idx": 11, "sample_idx": 1, "phase": "decode", "engine_step_id": 9},
        ])

    def test_prepare_decode_localization_keeps_flow_row_cap_out_of_global_decode_state(self) -> None:
        core = type("Core", (), {})()
        runtime = type("Runtime", (), {})()
        runtime.decode_row_idx = torch.zeros((3,), dtype=torch.long)
        runtime.decode_valid_mask = torch.zeros((3, 1), dtype=torch.float32)
        runtime.decode_prompt_idx_buf = torch.full((3,), -1, dtype=torch.long)
        runtime.decode_sample_idx_buf = torch.full((3,), -1, dtype=torch.long)
        runtime.decode_identity_staging = CudaLongHostStaging(
            3, torch.device("cpu"), channels=2
        )
        runtime.decode_logit_row_selector_staging = None
        runtime.decode_count = 0
        runtime.decode_prompt_idxs = []
        runtime.decode_sample_idxs = []
        runtime.decode_request_ids = []
        runtime.event_step_id = 1
        runtime.workload_execution = None
        core.RUNTIME = runtime
        core.pick_common_attn_metadata = staticmethod(lambda attn_metadata, spec_decode_common: attn_metadata)
        seen_max_rows: list[int] = []

        def _compute_decode_localization(**kwargs):
            seen_max_rows.append(int(kwargs["max_decode_rows"]))
            return torch.tensor([4, 5, 6], dtype=torch.long), [10, 11, 12], [0, 0, 0], [0, 1, 2]

        core.compute_decode_localization = staticmethod(_compute_decode_localization)
        core._resolve_prompt_sample_for_req_id = staticmethod(lambda req_id: (0, 0))

        runner = type("Runner", (), {})()
        runner.input_batch = type(
            "InputBatch",
            (),
            {
                "req_ids": ["reqA", "reqB", "reqC"],
                "num_reqs": 3,
                "req_id_to_index": {"reqA": 0, "reqB": 1, "reqC": 2},
                "num_prompt_tokens": [1, 1, 1],
                "num_computed_tokens_cpu": [1, 1, 1],
            },
        )()
        view = type(
            "View",
            (),
            {
                "logits_indices": torch.tensor([4, 5, 6], dtype=torch.long),
                "attn_metadata": type("Common", (), {"num_actual_tokens": 8})(),
                "spec_decode_common": None,
            },
        )()

        port_runtime_hooks.prepare_decode_localization(
            core=core,
            runner=runner,
            out=(None, None),
            prepare_inputs_view=view,
        )

        self.assertEqual(runtime.decode_count, 3)
        self.assertEqual(runtime.decode_request_ids, ["reqA", "reqB", "reqC"])
        self.assertEqual(runtime.decode_prompt_idxs, [10, 11, 12])
        self.assertEqual(runtime.decode_sample_idxs, [0, 0, 0])
        self.assertEqual(tuple(runtime.decode_row_idx.tolist()), (4, 5, 6))
        self.assertEqual(tuple(runtime.decode_valid_mask[:, 0].tolist()), (1.0, 1.0, 1.0))
        self.assertEqual(
            tuple(runtime.decode_prompt_idx_tensor.tolist()), (10, 11, 12)
        )
        self.assertEqual(
            tuple(runtime.decode_sample_idx_tensor.tolist()), (0, 0, 0)
        )
        self.assertEqual(seen_max_rows, [0])


if __name__ == "__main__":
    unittest.main()

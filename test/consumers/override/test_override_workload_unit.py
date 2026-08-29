"""Contracts for the typed native OverRIDE workload."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.contracts.hidden_locator import HiddenLocatorCapability
from tllm.consumers.override.workload import (
    OverRIDEConfig,
    OverRIDEWorkload,
    iteration_from_request_id,
)
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime import residual_runtime
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState

_HIDDEN = 6
_VOCAB = 11


def _spec() -> RuntimeSpecializationSpec:
    return RuntimeSpecializationSpec(
        rows=2,
        hidden_size=_HIDDEN,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _timeline() -> object:
    capabilities = BackendCapabilities(
        backend_id="vllm-0.27-mrv2",
        runner_family="mrv2",
        sampler_stages=frozenset(),
        native_min_p=False,
        tuple_sampler_return=False,
        split_execute_and_sample=True,
        graph_safety=GraphSafety.EAGER_ONLY,
        hidden_locators=frozenset({HiddenLocatorCapability.BLOCK_OUTPUT_DECODE}),
    )
    return export_main_workload_timeline(
        capabilities, graph_mode="eager", static_headroom_bytes=1 << 20
    )


def _workload() -> OverRIDEWorkload:
    return OverRIDEWorkload(
        OverRIDEConfig(
            hidden_size=_HIDDEN,
            vocab_size=_VOCAB,
            rank=2,
            num_iterations=4,
            learning_rate=0.05,
            source_layer_path="model.norm",
        )
    )


def _view(hidden: torch.Tensor) -> SamplerStepView:
    return SamplerStepView(
        engine_step_id=7,
        phase="decode",
        logits=torch.randn(2, _VOCAB),
        sampling_metadata=object(),
        decode_count=2,
        request_ids=("0_req", "1_req"),
        prompt_idxs=(0, 0),
        sample_idxs=(0, 1),
        prompt_idx_tensor=None,
        sample_idx_tensor=None,
        source_hidden=hidden,
        device=torch.device("cpu"),
        model=torch.nn.Module(),
        runner=object(),
    )


class OverRIDEWorkloadUnitTest(unittest.TestCase):
    def test_iteration_parser_is_algorithm_local(self) -> None:
        self.assertEqual(iteration_from_request_id("0_request"), 1)
        self.assertEqual(iteration_from_request_id("7_request"), 8)
        self.assertEqual(iteration_from_request_id("request"), 1)

    def test_config_constraints_are_rust_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "constraint"):
            OverRIDEConfig(rank=0)
        with self.assertRaisesRegex(ValueError, "constraint"):
            OverRIDEConfig(vocab_size=0)

    def test_cold_specialization_initializes_all_heads(self) -> None:
        workload = _workload()
        workload.cold_specialize(_spec())
        self.assertEqual(tuple(workload.heads.down.shape), (3, 2, _HIDDEN))
        self.assertEqual(tuple(workload.heads.up.shape), (3, _VOCAB, 2))

    def test_declaration_uses_signed_sampler_ports_without_callbacks(self) -> None:
        workload = _workload()
        runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=1 << 20)
        runtime.cold_specialize_workloads(_spec())
        compilation = runtime.compile_workloads(_timeline())
        tasks = tuple(
            task
            for region in compilation.decision.execution_regions
            for task in region.tasks
        )
        self.assertEqual(
            tasks,
            (
                "select_heads",
                "reweight",
                "train",
                "train__optimize_heads",
            ),
        )
        declaration = workload.declaration()
        inputs = {
            task.name: tuple(value.kind.value for value in task.inputs)
            for task in declaration.ir.tasks
        }
        self.assertEqual(inputs["select_heads"], ("sampler_request_ids",))
        self.assertIn("sampler_logits", inputs["train"])
        self.assertIn("sampled_tokens", inputs["train"])

    def test_signed_sampler_and_training_step_updates_heads(self) -> None:
        workload = _workload()
        runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=1 << 20)
        runtime.cold_specialize_workloads(_spec())
        compilation = runtime.compile_workloads(_timeline())
        hidden = torch.randn(2, _HIDDEN)
        state = ResidualRuntimeState()
        state.decode_count = 2
        state.decode_request_ids = ["0_req", "1_req"]
        state.decode_prompt_idxs = [0, 0]
        state.decode_sample_idxs = [0, 1]
        locator = ResidualLocator(0, "block_output", "decode")
        state.residual_bindings = {"source": ResidualPathBinding(locator, "source")}
        state.tap_decode_hidden = {"source": hidden}
        before = workload.heads.up.detach().clone()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation, bindings=runtime.workload_execution_bindings
            )
            residual_runtime.prepare_physical_workgraph_step(7)
            regions = compilation.decision.execution_regions
            pre_draw = regions[0].placement_anchor
            residual_runtime.advance_workload_anchor(pre_draw)
            provider = workload.native_workload_resources().sampler_provider
            assert provider is not None
            view = _view(hidden)
            modifier = provider.prepare_step(view)
            self.assertIsNotNone(modifier)
            provider.after_sample(view, torch.tensor([2, 3]))
            for region in regions:
                if region.placement_anchor != pre_draw:
                    residual_runtime.advance_workload_anchor(region.placement_anchor)

        self.assertEqual(workload.state_version, 1)
        self.assertFalse(torch.equal(before, workload.heads.up))


if __name__ == "__main__":
    unittest.main()

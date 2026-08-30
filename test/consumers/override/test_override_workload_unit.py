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
)
from tllm.kit.sampler_output import DeclaredSamplerOutput
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime import residual_runtime
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_bindings import (
    LocalRuntimeServices,
    SamplerRuntimeBinding,
)
from tllm.schedule import MainWorkloadTimeline

_HIDDEN = 6
_VOCAB = 11


def _spec() -> RuntimeSpecializationSpec:
    return RuntimeSpecializationSpec(
        rows=2,
        hidden_size=_HIDDEN,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _timeline() -> MainWorkloadTimeline:
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


def _view(
    hidden: torch.Tensor,
    *,
    sample_idxs: tuple[int, int] = (0, 1),
    engine_step_id: int = 7,
    logits: torch.Tensor | None = None,
) -> SamplerStepView:
    return SamplerStepView(
        engine_step_id=engine_step_id,
        phase="decode",
        logits=(
            torch.randn(2, _VOCAB)
            if logits is None
            else logits
        ),
        sampling_metadata=object(),
        decode_count=2,
        request_ids=("0_req", "1_req"),
        prompt_idxs=(0, 0),
        sample_idxs=sample_idxs,
        prompt_idx_tensor=None,
        sample_idx_tensor=None,
        source_hidden=hidden,
        device=torch.device("cpu"),
        model=torch.nn.Module(),
        runner=object(),
    )


class OverRIDEWorkloadUnitTest(unittest.TestCase):
    def test_typed_sample_indices_select_exact_head_domains(self) -> None:
        workload = _workload()

        previous, scales, train_rows, train_heads = workload.select_heads(
            (0, 1, 3)
        )

        torch.testing.assert_close(previous, torch.tensor((0, 0, 2)))
        torch.testing.assert_close(
            scales,
            torch.tensor(((0.0,), (0.8,), (0.8,))),
        )
        torch.testing.assert_close(train_rows, torch.tensor((0, 1)))
        torch.testing.assert_close(train_heads, torch.tensor((0, 1)))
        for invalid in ((-1,), (4,)):
            with self.assertRaisesRegex(RuntimeError, "configured iterations"):
                workload.select_heads(invalid)

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

    def test_single_iteration_has_no_heads_and_no_optimizer_transition(self) -> None:
        workload = OverRIDEWorkload(
            OverRIDEConfig(
                hidden_size=_HIDDEN,
                vocab_size=_VOCAB,
                rank=2,
                num_iterations=1,
            )
        )
        workload.cold_specialize(_spec())
        hidden = torch.randn(2, _HIDDEN)
        previous, scales, train_rows, train_heads = workload.select_heads((0, 0))

        delta, kept = workload.reweight(hidden, previous, scales)
        loss = workload.train(
            kept,
            train_rows,
            train_heads,
            torch.randn(2, _VOCAB),
            torch.tensor((1, 2)),
        )

        self.assertEqual(tuple(workload.heads.down.shape), (0, 2, _HIDDEN))
        torch.testing.assert_close(delta, torch.zeros(2, _VOCAB))
        self.assertIsNone(loss)
        self.assertEqual(workload.state_version, 0)

    def test_exact_reweight_delta_and_training_loss_match_tensor_oracle(self) -> None:
        workload = _workload()
        workload.cold_specialize(_spec())
        with torch.no_grad():
            workload.heads.down.copy_(
                torch.arange(
                    workload.heads.down.numel(),
                    dtype=torch.float32,
                ).reshape_as(workload.heads.down)
                / 50.0
            )
            workload.heads.up.copy_(
                torch.arange(
                    workload.heads.up.numel(),
                    dtype=torch.float32,
                ).reshape_as(workload.heads.up)
                / 70.0
            )
        hidden = torch.tensor(
            ((0.2, -0.5, 1.0, 0.7, -0.3, 0.4),
             (1.2, 0.3, -0.4, 0.1, 0.8, -0.2))
        )
        base_logits = torch.linspace(
            -1.0, 1.0, steps=2 * _VOCAB
        ).reshape(2, _VOCAB)
        tokens = torch.tensor((2, 3))
        previous, scales, train_rows, train_heads = workload.select_heads((0, 1))

        delta, kept = workload.reweight(hidden, previous, scales)
        loss = workload.train(
            kept,
            train_rows,
            train_heads,
            base_logits,
            tokens,
        )

        down0 = workload.heads.down[0]
        up0 = workload.heads.up[0]
        expected_delta = torch.zeros_like(base_logits)
        expected_delta[1] = (
            hidden[1] @ down0.transpose(0, 1) @ up0.transpose(0, 1)
        ) * workload.config.lambd
        selected_down = workload.heads.down.index_select(0, train_heads)
        selected_up = workload.heads.up.index_select(0, train_heads)
        expected_heads = torch.bmm(
            torch.bmm(
                hidden.unsqueeze(1),
                selected_down.transpose(1, 2),
            ),
            selected_up.transpose(1, 2),
        ).squeeze(1)
        expected_loss = torch.nn.functional.cross_entropy(
            base_logits + expected_heads,
            tokens,
        )
        torch.testing.assert_close(delta, expected_delta)
        assert loss is not None
        torch.testing.assert_close(loss, expected_loss)

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
        self.assertEqual(inputs["select_heads"], ("sampler_sample_idxs",))
        self.assertIn("sampler_logits", inputs["train"])
        self.assertIn("sampled_tokens", inputs["train"])

    def test_signed_sampler_and_training_step_updates_heads(self) -> None:
        workload = _workload()
        runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=1 << 20)
        runtime.cold_specialize_workloads(_spec())
        compilation = runtime.compile_workloads(_timeline())
        hidden = torch.randn(2, _HIDDEN)
        state = ResidualRuntimeState()
        state.event_step_id = 7
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
            sampler = workload.native_workload_resources().sampler
            assert isinstance(sampler, SamplerRuntimeBinding)
            assert isinstance(sampler.provider, DeclaredSamplerOutput)
            view = _view(hidden)
            sampler.stage_identity(7, view.request_ids, view.sample_idxs)
            residual_runtime.advance_workload_anchor(pre_draw)
            modifier = sampler.provider.prepare_step(view)
            self.assertIsNotNone(modifier)
            sampler.provider.after_sample(view, torch.tensor([2, 3]))
            for region in regions:
                if region.placement_anchor != pre_draw:
                    residual_runtime.advance_workload_anchor(region.placement_anchor)

        self.assertEqual(workload.state_version, 1)
        self.assertFalse(torch.equal(before, workload.heads.up))

    def test_two_signed_steps_match_adam_parameters_and_moments(self) -> None:
        workload = _workload()
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(_spec())
        compilation = runtime.compile_workloads(_timeline())
        expected_down = workload.heads.down.detach().clone()
        expected_up = workload.heads.up.detach().clone()
        down_exp_avg = torch.zeros_like(expected_down)
        up_exp_avg = torch.zeros_like(expected_up)
        down_exp_avg_sq = torch.zeros_like(expected_down)
        up_exp_avg_sq = torch.zeros_like(expected_up)
        reference_optimizer = torch.optim.Adam(
            (
                torch.nn.Parameter(expected_down.clone()),
                torch.nn.Parameter(expected_up.clone()),
            ),
            lr=workload.config.learning_rate,
            foreach=False,
        )
        steps = (
            (
                torch.tensor(
                    ((0.2, -0.5, 1.0, 0.7, -0.3, 0.4),
                     (1.2, 0.3, -0.4, 0.1, 0.8, -0.2))
                ),
                torch.linspace(-1.0, 1.0, steps=2 * _VOCAB).reshape(2, _VOCAB),
                torch.tensor((2, 3)),
            ),
            (
                torch.tensor(
                    ((-0.1, 0.6, 0.2, -0.8, 0.5, 1.1),
                     (0.9, -0.7, 0.4, 0.3, -0.2, 0.6))
                ),
                torch.linspace(0.7, -0.9, steps=2 * _VOCAB).reshape(2, _VOCAB),
                torch.tensor((4, 1)),
            ),
        )

        for step_id, (hidden, base_logits, tokens) in enumerate(steps, start=1):
            activation = hidden @ expected_down[0].transpose(0, 1)
            logits = base_logits + activation @ expected_up[0].transpose(0, 1)
            logits_gradient = torch.softmax(logits, dim=-1)
            logits_gradient[torch.arange(2), tokens] -= 1.0
            logits_gradient /= 2.0
            up_gradient = torch.zeros_like(expected_up)
            down_gradient = torch.zeros_like(expected_down)
            up_gradient[0] = logits_gradient.transpose(0, 1) @ activation
            down_gradient[0] = (
                logits_gradient @ expected_up[0]
            ).transpose(0, 1) @ hidden
            down_exp_avg.mul_(0.9).add_(down_gradient, alpha=0.1)
            up_exp_avg.mul_(0.9).add_(up_gradient, alpha=0.1)
            down_exp_avg_sq.mul_(0.999).addcmul_(
                down_gradient, down_gradient, value=0.001
            )
            up_exp_avg_sq.mul_(0.999).addcmul_(
                up_gradient, up_gradient, value=0.001
            )
            step_size = workload.config.learning_rate / (1.0 - 0.9**step_id)
            bias_correction = (1.0 - 0.999**step_id) ** 0.5
            with torch.no_grad():
                expected_down.addcdiv_(
                    down_exp_avg,
                    down_exp_avg_sq.sqrt() / bias_correction + 1e-8,
                    value=-step_size,
                )
                expected_up.addcdiv_(
                    up_exp_avg,
                    up_exp_avg_sq.sqrt() / bias_correction + 1e-8,
                    value=-step_size,
                )

            state = ResidualRuntimeState()
            state.event_step_id = step_id
            state.decode_count = 2
            state.decode_request_ids = ["req-a", "req-b"]
            state.decode_prompt_idxs = [0, 0]
            state.decode_sample_idxs = [0, 0]
            locator = ResidualLocator(0, "block_output", "decode")
            state.residual_bindings = {
                "source": ResidualPathBinding(locator, "source")
            }
            state.tap_decode_hidden = {"source": hidden}
            with mock.patch.object(residual_runtime, "RUNTIME", state):
                residual_runtime.publish_workload_compilation(
                    compilation, bindings=runtime.workload_execution_bindings
                )
                residual_runtime.prepare_physical_workgraph_step(step_id)
                regions = compilation.decision.execution_regions
                pre_draw = regions[0].placement_anchor
                sampler = workload.native_workload_resources().sampler
                assert isinstance(sampler, SamplerRuntimeBinding)
                assert isinstance(sampler.provider, DeclaredSamplerOutput)
                view = SamplerStepView(
                    engine_step_id=step_id,
                    phase="decode",
                    logits=base_logits,
                    sampling_metadata=object(),
                    decode_count=2,
                    request_ids=("req-a", "req-b"),
                    prompt_idxs=(0, 0),
                    sample_idxs=(0, 0),
                    prompt_idx_tensor=None,
                    sample_idx_tensor=None,
                    source_hidden=hidden,
                    device=torch.device("cpu"),
                    model=torch.nn.Module(),
                    runner=object(),
                )
                sampler.stage_identity(
                    step_id, view.request_ids, view.sample_idxs
                )
                residual_runtime.advance_workload_anchor(pre_draw)
                self.assertIsNotNone(sampler.provider.prepare_step(view))
                sampler.provider.after_sample(view, tokens)
                for region in regions:
                    if region.placement_anchor != pre_draw:
                        residual_runtime.advance_workload_anchor(
                            region.placement_anchor
                        )

            self.assertEqual(workload.state_version, step_id)
            torch.testing.assert_close(workload.heads.down, expected_down)
            torch.testing.assert_close(workload.heads.up, expected_up)

        expected_optimizer_state = reference_optimizer.state_dict()
        expected_optimizer_state["state"] = {
            0: {
                "step": torch.tensor(2.0),
                "exp_avg": down_exp_avg,
                "exp_avg_sq": down_exp_avg_sq,
            },
            1: {
                "step": torch.tensor(2.0),
                "exp_avg": up_exp_avg,
                "exp_avg_sq": up_exp_avg_sq,
            },
        }
        torch.testing.assert_close(
            workload.optimizer.state_dict(),
            expected_optimizer_state,
        )
        services = workload.native_workload_resources().services
        assert isinstance(services, LocalRuntimeServices)
        self.assertEqual(services.stats(False).loss_count, 2)

    def test_final_iteration_reweights_without_nan_or_state_update(self) -> None:
        workload = _workload()
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(_spec())
        compilation = runtime.compile_workloads(_timeline())
        hidden = torch.randn(2, _HIDDEN)
        state = ResidualRuntimeState()
        state.event_step_id = 8
        state.decode_count = 2
        state.decode_request_ids = ["0_req", "1_req"]
        state.decode_prompt_idxs = [0, 0]
        state.decode_sample_idxs = [3, 3]
        locator = ResidualLocator(0, "block_output", "decode")
        state.residual_bindings = {
            "source": ResidualPathBinding(locator, "source")
        }
        state.tap_decode_hidden = {"source": hidden}
        before_down = workload.heads.down.detach().clone()
        before_up = workload.heads.up.detach().clone()

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation, bindings=runtime.workload_execution_bindings
            )
            residual_runtime.prepare_physical_workgraph_step(8)
            regions = compilation.decision.execution_regions
            pre_draw = regions[0].placement_anchor
            sampler = workload.native_workload_resources().sampler
            assert isinstance(sampler, SamplerRuntimeBinding)
            assert isinstance(sampler.provider, DeclaredSamplerOutput)
            view = _view(hidden, sample_idxs=(3, 3), engine_step_id=8)
            sampler.stage_identity(8, view.request_ids, view.sample_idxs)
            residual_runtime.advance_workload_anchor(pre_draw)
            modifier = sampler.provider.prepare_step(view)
            self.assertIsNotNone(modifier)
            sampler.provider.after_sample(view, torch.tensor((2, 3)))
            for region in regions:
                if region.placement_anchor != pre_draw:
                    residual_runtime.advance_workload_anchor(
                        region.placement_anchor
                    )

        services = workload.native_workload_resources().services
        assert isinstance(services, LocalRuntimeServices)
        self.assertEqual(workload.state_version, 0)
        self.assertEqual(services.stats(False).loss_count, 0)
        torch.testing.assert_close(workload.heads.down, before_down)
        torch.testing.assert_close(workload.heads.up, before_up)


if __name__ == "__main__":
    unittest.main()

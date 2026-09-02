"""Contracts for the typed native TTRL workload."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.consumers.ttrl.workload import (
    TTRLConfig,
    TTRLWorkload,
    prepare_ttrl_training_batch,
    stage_ttrl_training,
)
from tllm.consumers.ttrl.rewards import (
    TTRLSample,
    build_ttrl_reward_result,
)
from tllm.consumers.ttrl.verifier import BoxedNormalizedVerifier
from tllm.runtime import residual_runtime
from tllm.runtime.generation_trace_runtime import (
    CompletionTraceEntry,
    GenerationTraceBatch,
)
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_bindings import LocalRuntimeServices
from tllm.schedule import MainWorkloadTimeline, WorkloadCompilation

_HIDDEN = 8


def _trace() -> GenerationTraceBatch:
    texts = ("work \\boxed{4}", "again \\boxed{4}", "other \\boxed{5}")
    return GenerationTraceBatch(
        entries=tuple(
            CompletionTraceEntry(
                request_id=f"req-{index}",
                prompt_idx=0,
                sample_idx=index,
                output_index=0,
                token_ids=(1, 2),
                text=text,
                finish_reason="stop",
                group_id="g0",
            )
            for index, text in enumerate(texts)
        )
    )


def _spec() -> RuntimeSpecializationSpec:
    return RuntimeSpecializationSpec(
        rows=3,
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
        hidden_locators=frozenset(),
    )
    return export_main_workload_timeline(
        capabilities, graph_mode="eager", static_headroom_bytes=1 << 20
    )


def _compile(
    workload: TTRLWorkload,
) -> tuple[TllmRuntime[TTRLWorkload], WorkloadCompilation]:
    runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=1 << 20)
    runtime.cold_specialize_workloads(_spec())
    return runtime, runtime.compile_workloads(_timeline())


def _drive(
    runtime: TllmRuntime[TTRLWorkload],
    compilation: WorkloadCompilation,
    step_id: int = 7,
) -> int:
    state = ResidualRuntimeState()
    with mock.patch.object(residual_runtime, "RUNTIME", state):
        residual_runtime.publish_workload_compilation(
            compilation, bindings=runtime.workload_execution_bindings
        )
        residual_runtime.prepare_physical_workgraph_step(step_id)
        for region in compilation.decision.execution_regions:
            residual_runtime.advance_workload_anchor(region.placement_anchor)
        execution = state.workload_execution
        assert execution is not None
        return execution.state_versions.observed_version("reward_model", None)


class TTRLWorkloadUnitTest(unittest.TestCase):
    def test_signed_step_trains_and_returns_versioned_control(self) -> None:
        workload = TTRLWorkload(
            TTRLConfig(hidden_size=_HIDDEN, rank=2, learning_rate=0.05)
        )
        runtime, compilation = _compile(workload)
        batch = prepare_ttrl_training_batch(
            _trace(),
            torch.randn(3, _HIDDEN),
            consumer_id=workload.config.consumer_id,
        )
        transaction = stage_ttrl_training(runtime, batch)
        before = tuple(parameter.detach().clone() for parameter in workload.reward_model.parameters())

        self.assertEqual(_drive(runtime, compilation), 1)
        after = tuple(parameter.detach() for parameter in workload.reward_model.parameters())
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))
        )
        published = transaction.commit()
        self.assertEqual(published.adapter_version_id, "ttrl:update:1")
        self.assertTrue(
            all(
                entry.adapter_version_id == "ttrl:update:1"
                for entry in published.retry_schedule
            )
        )
        self.assertEqual(
            tuple(entry.output_index for entry in published.retry_schedule),
            (0,),
        )
        self.assertEqual(runtime.drain(TTRLWorkload.control), ())
        self.assertEqual(runtime.drain(TTRLWorkload.reward_outcomes), ())
        self.assertEqual(runtime.drain(TTRLWorkload.retry_controls), ())
        self.assertEqual(runtime.drain(TTRLWorkload.update_barriers), ())
        with self.assertRaisesRegex(RuntimeError, "already closed"):
            transaction.commit()
        services = workload.native_workload_resources().services
        assert isinstance(services, LocalRuntimeServices)
        self.assertEqual(services.stats(False).loss_count, 1)

    def test_two_signed_updates_match_adam_reference(self) -> None:
        workload = TTRLWorkload(
            TTRLConfig(hidden_size=_HIDDEN, rank=2, learning_rate=0.05)
        )
        runtime, compilation = _compile(workload)
        expected_down = workload.reward_model.down.detach().clone()
        expected_up = workload.reward_model.up.detach().clone()
        down_exp_avg = torch.zeros_like(expected_down)
        up_exp_avg = torch.zeros_like(expected_up)
        down_exp_avg_sq = torch.zeros_like(expected_down)
        up_exp_avg_sq = torch.zeros_like(expected_up)
        hidden_steps = (
            torch.linspace(-1.0, 1.0, steps=3 * _HIDDEN).reshape(
                3, _HIDDEN
            ),
            torch.linspace(0.7, -0.9, steps=3 * _HIDDEN).reshape(
                3, _HIDDEN
            ),
        )

        for step_id, hidden in enumerate(hidden_steps, start=1):
            batch = prepare_ttrl_training_batch(
                _trace(),
                hidden,
                consumer_id=workload.config.consumer_id,
            )
            target = torch.tensor(
                batch.intent.training_reward_values,
                dtype=torch.float32,
            ).reshape(-1, 1)
            activation = hidden @ expected_down.transpose(0, 1)
            prediction = activation @ expected_up.transpose(0, 1)
            prediction_gradient = 2.0 * (prediction - target) / hidden.shape[0]
            up_gradient = prediction_gradient.transpose(0, 1) @ activation
            down_gradient = (
                prediction_gradient @ expected_up
            ).transpose(0, 1) @ hidden
            down_exp_avg.mul_(0.9).add_(down_gradient, alpha=0.1)
            up_exp_avg.mul_(0.9).add_(up_gradient, alpha=0.1)
            down_exp_avg_sq.mul_(0.999).addcmul_(
                down_gradient, down_gradient, value=0.001
            )
            up_exp_avg_sq.mul_(0.999).addcmul_(
                up_gradient, up_gradient, value=0.001
            )
            step_size = 0.05 / (1.0 - 0.9**step_id)
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

            transaction = stage_ttrl_training(runtime, batch)
            self.assertEqual(
                _drive(runtime, compilation, step_id=step_id), step_id
            )
            published = transaction.commit()
            self.assertEqual(
                published.adapter_version_id,
                f"ttrl:update:{step_id}",
            )

        torch.testing.assert_close(workload.reward_model.down, expected_down)
        torch.testing.assert_close(workload.reward_model.up, expected_up)
        services = workload.native_workload_resources().services
        assert isinstance(services, LocalRuntimeServices)
        self.assertEqual(services.stats(False).loss_count, 2)

    def test_zero_trainable_rows_publish_base_control_without_update(self) -> None:
        workload = TTRLWorkload(TTRLConfig(hidden_size=_HIDDEN, rank=2))
        runtime, compilation = _compile(workload)
        trace = GenerationTraceBatch(
            entries=tuple(
                CompletionTraceEntry(
                    request_id=f"missing-{index}",
                    prompt_idx=0,
                    sample_idx=index,
                    output_index=0,
                    token_ids=(1,),
                    text="no boxed answer",
                    finish_reason="stop",
                    group_id="missing",
                )
                for index in range(2)
            )
        )
        batch = prepare_ttrl_training_batch(
            trace,
            torch.randn(2, _HIDDEN),
            consumer_id=workload.config.consumer_id,
        )
        transaction = stage_ttrl_training(runtime, batch)

        self.assertEqual(_drive(runtime, compilation), 0)
        published = transaction.commit()
        self.assertEqual(published.adapter_version_id, "ttrl:base")
        self.assertTrue(
            all(
                entry.reward is None
                for entry in published.reward_outcomes.entries
            )
        )
        self.assertEqual(
            tuple(entry.state for entry in published.update_barriers.entries),
            ("blocked",),
        )
        self.assertTrue(
            all(
                entry.action == "retry"
                for entry in published.retry_controls.entries
            )
        )

    def test_signed_prepare_selects_explicit_training_indices(self) -> None:
        workload = TTRLWorkload(TTRLConfig(hidden_size=_HIDDEN, rank=2))
        hidden = torch.arange(3 * _HIDDEN, dtype=torch.float32).reshape(
            3, _HIDDEN
        )
        batch = prepare_ttrl_training_batch(
            _trace(),
            hidden,
            consumer_id=workload.config.consumer_id,
            n_votes_per_prompt=3,
            n_samples_per_prompt=2,
        )

        selected, rewards, prepared_intent = workload.prepare(
            batch.hidden, batch.intent
        )

        self.assertEqual(prepared_intent.training_indices, (0, 1))
        torch.testing.assert_close(selected, hidden[:2])
        torch.testing.assert_close(rewards, torch.ones(2))

    def test_reward_tie_is_first_seen_and_nested_box_is_normalized(self) -> None:
        result = build_ttrl_reward_result(
            (
                TTRLSample(0, 0, "first \\boxed{5}"),
                TTRLSample(0, 1, "second \\boxed{4}"),
            )
        )
        self.assertEqual(result.majority_by_prompt[0].label, "5")
        self.assertEqual(result.rewards, [1.0, 0.0])
        verifier = BoxedNormalizedVerifier()
        self.assertEqual(
            verifier.label_for_completion(
                "work \\boxed{\\frac{1}{2}} trailing"
            ),
            "\\frac{1}{2}",
        )


if __name__ == "__main__":
    unittest.main()

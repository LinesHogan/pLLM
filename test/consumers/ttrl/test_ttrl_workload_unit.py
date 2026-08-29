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
)
from tllm.runtime import residual_runtime
from tllm.runtime.generation_trace_runtime import (
    CompletionTraceEntry,
    GenerationTraceBatch,
)
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState

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


def _timeline() -> object:
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


def _compile(workload: TTRLWorkload) -> tuple[TllmRuntime, object]:
    runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=1 << 20)
    runtime.cold_specialize_workloads(_spec())
    return runtime, runtime.compile_workloads(_timeline())


def _drive(runtime: TllmRuntime, compilation: object, step_id: int = 7) -> int:
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
    def test_config_constraints_are_rust_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "constraint"):
            TTRLConfig(rank=0)
        with self.assertRaisesRegex(ValueError, "constraint"):
            TTRLConfig(learning_rate=0.0)

    def test_declaration_orders_train_transition_publish(self) -> None:
        workload = TTRLWorkload(TTRLConfig(hidden_size=_HIDDEN, rank=2))
        _, compilation = _compile(workload)
        self.assertEqual(
            tuple(region.tasks for region in compilation.decision.execution_regions),
            (
                ("train",),
                ("train__optimize_reward_model",),
                ("publish",),
            ),
        )
        self.assertEqual(
            tuple(
                region.implementation_kind
                for region in compilation.decision.execution_regions
            ),
            ("compiled", "extern", "extern"),
        )

    def test_signed_step_trains_and_returns_versioned_control(self) -> None:
        workload = TTRLWorkload(
            TTRLConfig(hidden_size=_HIDDEN, rank=2, learning_rate=0.05)
        )
        runtime, compilation = _compile(workload)
        batch = prepare_ttrl_training_batch(_trace(), torch.randn(3, _HIDDEN))
        runtime.stage(TTRLWorkload.train_hidden, batch.hidden)
        runtime.stage(TTRLWorkload.train_rewards, batch.rewards)
        runtime.stage(TTRLWorkload.pending_intent, batch.intent)
        before = tuple(parameter.detach().clone() for parameter in workload.reward_model.parameters())

        self.assertEqual(_drive(runtime, compilation), 1)
        after = tuple(parameter.detach() for parameter in workload.reward_model.parameters())
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))
        )
        (published,) = runtime.drain(TTRLWorkload.control)
        self.assertEqual(published.adapter_version_id, "ttrl:update:1")
        self.assertTrue(
            all(
                entry.adapter_version_id == "ttrl:update:1"
                for entry in published.retry_schedule
            )
        )
        stats_reader = workload.native_workload_resources().stats_reader
        assert stats_reader is not None
        self.assertEqual(stats_reader(False).loss_count, 1)

    def test_missing_staged_batch_skips_without_version_change(self) -> None:
        workload = TTRLWorkload(TTRLConfig(hidden_size=_HIDDEN, rank=2))
        runtime, compilation = _compile(workload)
        self.assertEqual(_drive(runtime, compilation), 0)
        self.assertEqual(runtime.drain(TTRLWorkload.control), ())

    def test_worker_reconstruction_keeps_typed_ports(self) -> None:
        workload = TTRLWorkload(TTRLConfig(hidden_size=_HIDDEN, rank=2))
        resources = workload.native_workload_resources()
        spec = resources.worker_spec
        assert spec is not None
        rebuilt = TTRLWorkload.reconstruct(spec.config)
        self.assertEqual(
            set(rebuilt.provided_inputs),
            {
                TTRLWorkload.train_hidden.key,
                TTRLWorkload.train_rewards.key,
                TTRLWorkload.pending_intent.key,
            },
        )
        self.assertEqual(set(rebuilt.output_sinks), {TTRLWorkload.control.key})


if __name__ == "__main__":
    unittest.main()

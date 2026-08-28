#!/usr/bin/env python3
"""Unit tests for the native TTRL Region workload (kit style).

Every training/publish path is driven through a full signed step:
compile -> publish compilation -> prepare -> advance anchors -> state-version
audit.  No helper may bypass the audit.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.consumers.ttrl.config import TTRLConsumerConfig
from tllm.consumers.ttrl.program import TTRLProgram
from tllm.consumers.ttrl.region_workload import TTRLRegionWorkload
from tllm.kit import clear_host_channels
from tllm.runtime import residual_runtime
from tllm.runtime.generation_trace_runtime import (
    CompletionTraceEntry,
    GenerationTraceBatch,
)
from tllm.runtime.retry_control_runtime import read_retry_controls
from tllm.runtime.reward_outcome_runtime import read_reward_outcomes
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.update_barrier_runtime import read_update_barriers
from tllm.runtime.workload_plugins import reconstruct_worker_bindings
from tllm.workload import declare_workload

_HIDDEN = 8
_RANK = 2
_LR = 0.05


def _config(**overrides: object) -> TTRLConsumerConfig:
    values: dict[str, object] = {
        "consumer_id": "ttrl",
        "hidden_size": _HIDDEN,
        "rank": _RANK,
        "learning_rate": _LR,
        "implementation": "region",
    }
    values.update(overrides)
    return TTRLConsumerConfig(**values)  # type: ignore[arg-type]


def _spec(rows: int = 4) -> RuntimeSpecializationSpec:
    return RuntimeSpecializationSpec(rows, _HIDDEN, torch.float32, torch.device("cpu"))


def _capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        backend_id="vllm-0.27-mrv2",
        runner_family="mrv2",
        sampler_stages=frozenset(),
        native_min_p=False,
        tuple_sampler_return=False,
        split_execute_and_sample=True,
        graph_safety=GraphSafety.EAGER_ONLY,
        hidden_locators=frozenset(),
    )


def _compile(workload: TTRLRegionWorkload) -> tuple[TllmRuntime, object]:
    runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=1 << 20)
    runtime.cold_specialize_workloads(_spec())
    compilation = runtime.compile_workloads(
        export_main_workload_timeline(
            _capabilities(), graph_mode="eager", static_headroom_bytes=1 << 20
        )
    )
    return runtime, compilation


def _drive_signed_step(runtime: TllmRuntime, compilation: object, step_id: int = 7):
    """Advance every signed Region once under a fresh residual runtime state."""
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
        version = execution.state_versions.observed_version("reward_model", None)
    return version


def _trace(texts: tuple[str, ...], *, group_id: str = "g0") -> GenerationTraceBatch:
    entries = tuple(
        CompletionTraceEntry(
            request_id=f"req-{index}",
            prompt_idx=0,
            sample_idx=index,
            output_index=0,
            token_ids=(1, 2),
            text=text,
            finish_reason="stop",
            group_id=group_id,
        )
        for index, text in enumerate(texts)
    )
    return GenerationTraceBatch(entries=entries)


_HAPPY_TEXTS = (
    "reasoning ... \\boxed{4}",
    "different path ... \\boxed{4}",
    "minority ... \\boxed{5}",
)


class _TTRLRegionTestBase(unittest.TestCase):
    def setUp(self) -> None:
        # The host channel registry is process-global (by design: the engine
        # drives a reconstructed instance, not the host object), so tests
        # sharing a consumer_id must not observe each other's staging state.
        clear_host_channels()


class TestTTRLRegionDeclaration(_TTRLRegionTestBase):
    def test_declaration_signs_three_ordered_regions(self) -> None:
        workload = TTRLRegionWorkload(_config())
        _, compilation = _compile(workload)

        regions = compilation.decision.execution_regions
        self.assertEqual(
            tuple(region.tasks for region in regions),
            (("train",), ("update",), ("publish",)),
        )
        self.assertEqual(
            tuple(region.implementation_kind for region in regions),
            ("compiled", "extern", "extern"),
        )
        self.assertEqual(len(compilation.decision.materializations), 1)
        # extern_io effect: the publish Region is a singleton, never merged,
        # never capture-safe, pinned to the default stream.
        publish = regions[2]
        self.assertFalse(publish.capture_safe)
        self.assertEqual(tuple(publish.tasks), ("publish",))

    def test_declaration_records_state_and_extern_io_effects(self) -> None:
        workload = TTRLRegionWorkload(_config())
        declared = declare_workload(workload)
        effects = {
            task.name: tuple(effect.kind for effect in task.effects)
            for task in declared.ir.tasks
        }
        self.assertEqual(effects["train"], ("read",))
        self.assertEqual(effects["update"], ("transition_maybe",))
        self.assertEqual(
            effects["publish"],
            ("read", "extern_io", "extern_io", "extern_io"),
        )

    def test_validate_config_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "implementation='region'"):
            TTRLRegionWorkload(_config(implementation="consumer"))
        with self.assertRaisesRegex(ValueError, "positive rank"):
            TTRLRegionWorkload(_config(rank=0))
        with self.assertRaisesRegex(ValueError, "positive learning_rate"):
            TTRLRegionWorkload(_config(learning_rate=0.0))
        with self.assertRaisesRegex(ValueError, "hidden_size >= 0"):
            TTRLRegionWorkload(_config(hidden_size=-1))

    def test_cold_specialize_resolves_and_checks_hidden_size(self) -> None:
        workload = TTRLRegionWorkload(_config(hidden_size=0))
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(_spec())
        down = workload.params["down"]
        self.assertIsInstance(down, torch.nn.Parameter)
        self.assertEqual(tuple(down.shape), (_RANK, _HIDDEN))

        mismatched = TTRLRegionWorkload(_config(hidden_size=99))
        with self.assertRaisesRegex(RuntimeError, "hidden_size"):
            mismatched.cold_specialize(_spec())


class TestTTRLRegionStaging(_TTRLRegionTestBase):
    def test_stage_aligns_rows_and_returns_intent(self) -> None:
        workload = TTRLRegionWorkload(_config())
        hidden_rows = torch.randn(3, _HIDDEN)

        intent = workload.stage_completed_generations(_trace(_HAPPY_TEXTS), hidden_rows)

        self.assertEqual(intent.training_row_count, 3)
        self.assertEqual(intent.training_reward_values, (1.0, 1.0, 0.0))
        self.assertEqual(workload.adapter_version_id(), "ttrl:base")
        staged = workload._train_hidden_for_step(0)
        self.assertTrue(torch.equal(staged, hidden_rows))
        staged_rewards = workload._train_rewards_for_step(0)
        self.assertTrue(
            torch.allclose(staged_rewards, torch.tensor([1.0, 1.0, 0.0]))
        )
        self.assertEqual(workload.resources.stats()["staged_reward_rows"], 3)

    def test_stage_selects_training_indices_when_rows_are_superset(self) -> None:
        # One vote per prompt over two samples: index selection mirrors the
        # legacy toy closed loop (rows keyed by training_indices).
        workload = TTRLRegionWorkload(_config())
        hidden_rows = torch.randn(2, _HIDDEN)

        intent = workload.stage_completed_generations(
            _trace(_HAPPY_TEXTS[:2]),
            hidden_rows,
            n_votes_per_prompt=1,
            n_samples_per_prompt=2,
        )

        self.assertEqual(intent.training_row_count, 2)
        self.assertEqual(intent.training_indices, (0, 1))

    def test_staging_fails_closed(self) -> None:
        workload = TTRLRegionWorkload(_config())
        trace = _trace(_HAPPY_TEXTS)
        hidden_rows = torch.randn(3, _HIDDEN)

        disabled = TTRLRegionWorkload(_config(enabled=False))
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            disabled.stage_completed_generations(trace, hidden_rows)

        with self.assertRaisesRegex(ValueError, "rank-2"):
            workload.stage_completed_generations(trace, torch.randn(3))
        with self.assertRaisesRegex(RuntimeError, "align"):
            workload.stage_completed_generations(trace, torch.randn(2, _HIDDEN))

        workload.stage_completed_generations(trace, hidden_rows)
        with self.assertRaisesRegex(RuntimeError, "already has a staged batch"):
            workload.stage_completed_generations(trace, hidden_rows)

    def test_unstaged_step_provides_empty_rows(self) -> None:
        workload = TTRLRegionWorkload(_config())
        workload.cold_specialize(_spec())
        hidden = workload._train_hidden_for_step(0)
        rewards = workload._train_rewards_for_step(0)
        self.assertEqual(tuple(hidden.shape), (0, _HIDDEN))
        self.assertEqual(tuple(rewards.shape), (0,))

    def test_publish_requires_bound_runtime(self) -> None:
        workload = TTRLRegionWorkload(_config())
        workload.cold_specialize(_spec())
        workload.stage_completed_generations(
            _trace(_HAPPY_TEXTS), torch.randn(3, _HIDDEN)
        )
        with self.assertRaisesRegex(RuntimeError, "bind_control_runtime"):
            workload.publish(7)

    def test_predict_scores_matches_current_params(self) -> None:
        workload = TTRLRegionWorkload(_config())
        workload.cold_specialize(_spec())
        hidden = torch.randn(3, _HIDDEN)
        down = workload.params["down"]
        up = workload.params["up"]
        assert isinstance(down, torch.nn.Parameter)
        assert isinstance(up, torch.nn.Parameter)
        expected = F.linear(F.linear(hidden, down), up)
        self.assertTrue(torch.allclose(workload.predict_scores(hidden), expected))


class TestTTRLRegionSignedSteps(_TTRLRegionTestBase):
    def test_happy_path_signed_step_trains_and_publishes(self) -> None:
        workload = TTRLRegionWorkload(_config())
        runtime, compilation = _compile(workload)
        control = types.SimpleNamespace()
        workload.bind_control_runtime(control)
        workload.stage_completed_generations(
            _trace(_HAPPY_TEXTS), torch.randn(3, _HIDDEN)
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in workload.params.items()
        }

        version = _drive_signed_step(runtime, compilation)

        self.assertEqual(version, 1)
        self.assertEqual(workload.state_version, 1)
        self.assertEqual(workload.adapter_version_id(), "ttrl:update:1")
        self.assertTrue(
            any(
                not torch.equal(before[name], workload.params[name].detach())
                for name in before
            )
        )
        stats = workload.resources.stats()
        self.assertEqual(stats["loss_count"], 1)
        self.assertEqual(stats["update_count"], 1)

        rewards = read_reward_outcomes(control)
        self.assertEqual(len(rewards.entries), 3)
        retries = read_retry_controls(control)
        self.assertEqual(
            tuple(entry.action for entry in retries.entries),
            ("accept", "accept", "retry"),
        )
        retry_entry = retries.entries[2]
        self.assertTrue(retry_entry.retry_after_update)
        barriers = read_update_barriers(control)
        self.assertEqual(len(barriers.entries), 1)
        self.assertEqual(barriers.entries[0].state, "ready")
        self.assertEqual(barriers.entries[0].ready_count, 3)

        published = workload.last_published_intent
        assert published is not None
        self.assertEqual(published.adapter_version_id, "ttrl:update:1")
        self.assertEqual(len(published.retry_schedule), 1)
        self.assertEqual(
            published.retry_schedule[0].adapter_version_id, "ttrl:update:1"
        )

    def test_zero_completion_step_bumps_no_version(self) -> None:
        workload = TTRLRegionWorkload(_config())
        runtime, compilation = _compile(workload)
        control = types.SimpleNamespace()
        workload.bind_control_runtime(control)

        version = _drive_signed_step(runtime, compilation)

        self.assertEqual(version, 0)
        self.assertEqual(workload.state_version, 0)
        self.assertEqual(len(read_reward_outcomes(control).entries), 0)
        self.assertEqual(len(read_retry_controls(control).entries), 0)
        self.assertEqual(len(read_update_barriers(control).entries), 0)
        self.assertIsNone(workload.last_published_intent)

    def test_zero_trainable_step_publishes_without_update(self) -> None:
        workload = TTRLRegionWorkload(_config())
        runtime, compilation = _compile(workload)
        control = types.SimpleNamespace()
        workload.bind_control_runtime(control)
        workload.stage_completed_generations(
            _trace(("no boxed answer at all",)), torch.randn(1, _HIDDEN)
        )

        version = _drive_signed_step(runtime, compilation)

        self.assertEqual(version, 0)
        self.assertEqual(workload.state_version, 0)
        self.assertEqual(workload.adapter_version_id(), "ttrl:base")
        stats = workload.resources.stats()
        self.assertEqual(stats["loss_count"], 0)
        self.assertEqual(stats["update_count"], 0)

        rewards = read_reward_outcomes(control)
        self.assertEqual(len(rewards.entries), 1)
        self.assertIsNone(rewards.entries[0].reward)
        barriers = read_update_barriers(control)
        self.assertEqual(barriers.entries[0].state, "blocked")
        published = workload.last_published_intent
        assert published is not None
        self.assertEqual(published.adapter_version_id, "ttrl:base")
        self.assertEqual(published.retry_schedule, ())

    def test_parameter_parity_with_legacy_program(self) -> None:
        torch.manual_seed(0)
        workload = TTRLRegionWorkload(_config())
        workload.cold_specialize(_spec())
        legacy = TTRLProgram(
            hidden_size=_HIDDEN,
            rank=_RANK,
            lr=_LR,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        with torch.no_grad():
            legacy.adapter.down.weight.copy_(workload.params["down"])
            legacy.adapter.up.weight.copy_(workload.params["up"])

        torch.manual_seed(1)
        hidden = torch.randn(3, _HIDDEN)
        rewards = torch.tensor([1.0, 1.0, 0.0])

        legacy_loss = legacy(hidden, rewards)
        legacy.optimizer.zero_grad(set_to_none=True)
        legacy_loss.backward()
        legacy.optimizer.step()

        native_loss = workload.train(hidden, rewards)
        workload.update(native_loss)

        self.assertTrue(torch.allclose(native_loss, legacy_loss, atol=1e-6))
        self.assertTrue(
            torch.allclose(
                workload.params["down"].detach(),
                legacy.adapter.down.weight,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                workload.params["up"].detach(),
                legacy.adapter.up.weight,
                atol=1e-6,
            )
        )
        self.assertEqual(workload.state_version, 1)

    def test_worker_spec_roundtrip(self) -> None:
        workload = TTRLRegionWorkload(_config())
        runtime, compilation = _compile(workload)
        worker_spec = workload.native_workload_resources().worker_spec
        self.assertIsNotNone(worker_spec)

        bindings = reconstruct_worker_bindings(
            compilation, (worker_spec,), specialization=_spec()
        )

        self.assertEqual(
            bindings.declarations[0].ir.semantic_key,
            declare_workload(workload).ir.semantic_key,
        )
        self.assertEqual(set(bindings.state_versions), {"reward_model"})

    def test_enable_epoch_rebuilds_resources_snapshot(self) -> None:
        workload = TTRLRegionWorkload(_config())
        first = workload.native_workload_resources()
        assert first.enabled_predicate is not None
        self.assertTrue(first.enabled_predicate())

        workload.set_enabled(False)

        second = workload.native_workload_resources()
        assert second.enabled_predicate is not None
        self.assertFalse(second.enabled_predicate())


if __name__ == "__main__":
    unittest.main()

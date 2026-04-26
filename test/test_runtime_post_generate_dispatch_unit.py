#!/usr/bin/env python3
"""Unit tests for post-generate flow dispatch."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from tllm.consumers.base import BaseConsumer
from tllm.contracts.port_bundle import PortBundle
from tllm.ports.base import ConsumerFlow
from tllm.ports.generation_trace import GenerationTrace
from tllm.ports.request_meta import RequestMeta
from tllm.ports.reward_outcome import RewardOutcome
from tllm.ports.retry_control import RetryControl
from tllm.ports.update_barrier import UpdateBarrier
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch
from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch, RewardOutcomeEntry, read_reward_outcomes
from tllm.runtime.retry_control_runtime import RetryControlBatch, RetryControlEntry, read_retry_controls
from tllm.runtime.update_barrier_runtime import UpdateBarrierBatch, UpdateBarrierEntry, read_update_barriers
from tllm.runtime.vllm_patch import port_runtime_hooks


class _PostGenerateConsumer(BaseConsumer):
    def __init__(self) -> None:
        self.seen: list[PortBundle] = []
        self.feedback_calls = 0

    @property
    def consumer_id(self) -> str:
        return "post_generate_consumer"

    def flows(self):
        return (
            ConsumerFlow(
                reads=(GenerationTrace.read(),),
                writes=(RewardOutcome.write(),),
                window="out_of_band",
                bundle_key=("generation_group_id", "phase"),
            ),
        )

    def consume_bundle(self, bundle, ctx) -> None:
        self.seen.append(bundle)
        trace = bundle.entries["generation_trace"]
        sink = bundle.entries["reward_outcome"]
        first = trace.entries[0]
        sink.append(
            RewardOutcomeBatch(
                entries=(
                    RewardOutcomeEntry(
                        request_id=first.request_id,
                        prompt_idx=first.prompt_idx,
                        sample_idx=first.sample_idx,
                        output_index=first.output_index,
                        group_id=first.group_id,
                        reward=1.0,
                        judge="toy",
                        label="accepted",
                        is_correct=True,
                    ),
                )
            )
        )

    def on_step_end(self, ctx) -> None:
        _ = ctx
        self.feedback_calls += 1


class _RetryBarrierConsumer(BaseConsumer):
    @property
    def consumer_id(self) -> str:
        return "retry_barrier_consumer"

    def flows(self):
        return (
            ConsumerFlow(
                reads=(GenerationTrace.read(),),
                writes=(RetryControl.write(), UpdateBarrier.write()),
                window="out_of_band",
                bundle_key=("generation_group_id", "phase"),
            ),
        )

    def consume_bundle(self, bundle, ctx) -> None:
        _ = ctx
        trace = bundle.entries["generation_trace"]
        first = trace.entries[0]
        bundle.entries["retry_control"].append(
            RetryControlBatch(
                entries=(
                    RetryControlEntry(
                        request_id=first.request_id,
                        prompt_idx=first.prompt_idx,
                        sample_idx=first.sample_idx,
                        output_index=first.output_index,
                        group_id=first.group_id,
                        action="retry",
                        reason="toy retry",
                        retry_after_update=True,
                    ),
                )
            )
        )
        bundle.entries["update_barrier"].append(
            UpdateBarrierBatch(
                entries=(
                    UpdateBarrierEntry(
                        barrier_id="barrier-round-a",
                        group_id=first.group_id,
                        consumer_id=self.consumer_id,
                        state="ready",
                        request_ids=(first.request_id,),
                        ready_count=1,
                    ),
                )
            )
        )


class PostGenerateDispatchUnitTest(unittest.TestCase):
    def _runtime(self):
        runtime = SimpleNamespace()
        runtime.event_step_id = 17
        runtime.last_generation_trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-a",
                    prompt_idx=3,
                    sample_idx=2,
                    output_index=0,
                    token_ids=(1, 2, 3),
                    text="answer",
                    finish_reason="stop",
                    group_id="round-a",
                ),
            )
        )
        return runtime

    def test_post_generate_flow_reads_trace_and_writes_reward_outcome(self) -> None:
        runtime = self._runtime()
        consumer = _PostGenerateConsumer()
        runtime.dispatch_plan = DispatchPlan.build([consumer])
        runner = SimpleNamespace(device=torch.device("cpu"), model=None, use_cuda_graph=False, compilation_config=None)

        dispatched = port_runtime_hooks.dispatch_post_generate_port_bundles(runtime=runtime, runner=runner)

        self.assertEqual(dispatched, 1)
        self.assertEqual(len(consumer.seen), 1)
        self.assertEqual(consumer.feedback_calls, 1)
        bundle = consumer.seen[0]
        self.assertEqual(bundle.key.phase, "post_generate")
        self.assertEqual(bundle.key.engine_step_id, 17)
        self.assertIs(bundle.entries["generation_trace"], runtime.last_generation_trace)
        outcomes = read_reward_outcomes(runtime)
        self.assertEqual(len(outcomes.entries), 1)
        self.assertEqual(outcomes.entries[0].request_id, "req-a")
        self.assertEqual(outcomes.entries[0].reward, 1.0)

    def test_reward_outcome_sink_rejects_non_batch_payloads(self) -> None:
        runtime = self._runtime()
        flow = ConsumerFlow(
            reads=(GenerationTrace.read(),),
            writes=(RewardOutcome.write(),),
            window="out_of_band",
            bundle_key=("generation_group_id", "phase"),
        )

        bundle = port_runtime_hooks.build_post_generate_port_bundle(runtime=runtime, flow=flow)
        with self.assertRaisesRegex(TypeError, "batch must be RewardOutcomeBatch"):
            bundle.entries["reward_outcome"].append("bad")  # type: ignore[attr-defined]

    def test_post_generate_flow_writes_retry_control_and_update_barrier(self) -> None:
        runtime = self._runtime()
        consumer = _RetryBarrierConsumer()
        runtime.dispatch_plan = DispatchPlan.build([consumer])
        runner = SimpleNamespace(device=torch.device("cpu"), model=None, use_cuda_graph=False, compilation_config=None)

        dispatched = port_runtime_hooks.dispatch_post_generate_port_bundles(runtime=runtime, runner=runner)

        self.assertEqual(dispatched, 1)
        retry_entries = read_retry_controls(runtime).entries
        barrier_entries = read_update_barriers(runtime).entries
        self.assertEqual(len(retry_entries), 1)
        self.assertEqual(retry_entries[0].action, "retry")
        self.assertTrue(retry_entries[0].retry_after_update)
        self.assertEqual(len(barrier_entries), 1)
        self.assertEqual(barrier_entries[0].state, "ready")
        self.assertEqual(barrier_entries[0].request_ids, ("req-a",))

    def test_retry_and_barrier_sinks_reject_wrong_payload_types(self) -> None:
        runtime = self._runtime()
        flow = ConsumerFlow(
            reads=(GenerationTrace.read(),),
            writes=(RetryControl.write(), UpdateBarrier.write()),
            window="out_of_band",
            bundle_key=("generation_group_id", "phase"),
        )

        bundle = port_runtime_hooks.build_post_generate_port_bundle(runtime=runtime, flow=flow)
        with self.assertRaisesRegex(TypeError, "batch must be RetryControlBatch"):
            bundle.entries["retry_control"].append("bad")  # type: ignore[attr-defined]
        with self.assertRaisesRegex(TypeError, "batch must be UpdateBarrierBatch"):
            bundle.entries["update_barrier"].append("bad")  # type: ignore[attr-defined]

    def test_post_generate_reward_write_mode_must_be_explicit_string(self) -> None:
        from tllm.ports.base import PortKind, PortWrite
        from tllm.ports.reward_outcome import RewardOutcomeLocator

        runtime = self._runtime()
        flow = ConsumerFlow(
            reads=(GenerationTrace.read(),),
            writes=(PortWrite(kind=PortKind.REWARD_OUTCOME, locator=RewardOutcomeLocator(), mode=object()),),  # type: ignore[arg-type]
            window="out_of_band",
            bundle_key=("generation_group_id", "phase"),
        )

        with self.assertRaisesRegex(TypeError, "reward outcome write mode must be str"):
            port_runtime_hooks.build_post_generate_port_bundle(runtime=runtime, flow=flow)

    def test_decode_dispatch_ignores_post_generate_only_flows(self) -> None:
        runtime = SimpleNamespace()
        runtime.dispatch_plan = DispatchPlan.build([_PostGenerateConsumer()])
        runtime.event_step_id = 1
        runtime.decode_count = 1
        runtime.tap_decode_hidden = {}
        runtime.tap_decode_hidden_compact = {}
        runtime.decode_compact_count = 0
        runtime.decode_compact_row_ids = ()
        runtime.decode_request_ids = ["req-a"]
        runtime.decode_prompt_idxs = [0]
        runtime.decode_sample_idxs = [0]
        runtime.residual_bindings = {}
        core = SimpleNamespace(RUNTIME=runtime)
        runner = SimpleNamespace(device=torch.device("cpu"), model=None, use_cuda_graph=False, compilation_config=None)

        with unittest.mock.patch.object(
            port_runtime_hooks, "build_decode_port_frames", side_effect=AssertionError("decode frames should not be built")
        ):
            dispatched = port_runtime_hooks.dispatch_decode_port_bundles(core=core, runner=runner)

        self.assertEqual(dispatched, 0)

    def test_generate_wrapper_dispatches_post_generate_flows(self) -> None:
        from test.test_runtime_generation_trace_unit import _Candidate, _LLM, _Output
        from tllm.runtime.vllm_patch.capture_runner import run_generate_with_request_mapping

        runtime = SimpleNamespace()
        consumer = _PostGenerateConsumer()
        runtime.dispatch_plan = DispatchPlan.build([consumer])
        llm = _LLM([_Output("req0", [_Candidate([9], text="done", finish_reason="stop")])])

        run_generate_with_request_mapping(
            runtime=runtime,
            llm=llm,
            prompts=["prompt"],
            params=[object()],
            request_prompt_indices=[5],
            generation_group_id="round-b",
        )

        self.assertEqual(len(consumer.seen), 1)
        self.assertEqual(read_reward_outcomes(runtime).entries[0].group_id, "round-b")


if __name__ == "__main__":
    unittest.main()

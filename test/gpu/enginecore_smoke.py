"""Workloads in a real vLLM EngineCore child process, with the frontend never touching CUDA.

Covers ESamp (eager and captured), shared completions feeding FTTT/TTRL, ReFT and
Output LoRA inputs, automatic feedback retries, TLM, Dummy exports, a user workload
loaded by reference and CUDA inputs shared over IPC. Run with a flag from MODES to
execute one scenario.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from huggingface_hub import snapshot_download
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

import pllm
from pllm.authoring import CausalTrainBatch, StateTensorUpdate, WeightedCEBatch
from pllm.config import ExecutionOptions, ModelConfig, RunnerName
from pllm.contracts.backend_attachment import (
    ActivationPatchOperator,
)
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import StateScope
from pllm.execution.trace import TracePhase, TraceSelection
from pllm.host import HostConfig, make_host
from pllm.workloads.dummy import DummyConfig, DummyWorkload
from pllm.workloads.esamp_request import ESampRequestConfig, ESampRequestWorkload
from pllm.workloads.fttt import (
    FTTTConfig,
    FTTTDefaultJudge,
    FTTTFeedbackService,
    FTTTJudgeResult,
    FTTTJudgeService,
    FTTTWorkload,
)
from pllm.workloads.fttt.types import FTTTResult
from pllm.workloads.lora import OutputLoRAConfig, OutputLoRAWorkload
from pllm.workloads.prefill_adapters import PrefillAdapters
from pllm.workloads.recirculation import RecirculationWorkload
from pllm.workloads.reft import ActivationConfig, ActivationWorkload
from pllm.workloads.tlm import TLMConfig, TLMWorkload
from pllm.workloads.ttrl import TTRLConfig, TTRLWorkload, ttrl_rewards

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = ["The sky is", "Water freezes at"]


def engine(model=MODEL, seqs=2, tokens=128, memory=0.25, cuda_graphs=False, **options) -> ModelConfig:
    """A small EngineCore deployment: max_model_len 64, bfloat16, seed 7; eager unless an arm asks for graphs."""
    return ModelConfig(
        model, "bfloat16", 64, seqs, tokens, memory, seed=7, multiprocessing=True, cuda_graphs=cuda_graphs, **options
    )


def greedy(max_tokens, **options) -> SamplingParams:
    return SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True, **options)


def assert_clean(diagnostics) -> None:
    assert diagnostics.active_lineages == diagnostics.inference_pins == diagnostics.pending_state_releases == 0


class RetryOnce(FTTTJudgeService):
    """Prompt 7 is right at once; every other prompt is right on its first retry."""

    def judge(self, completion):
        correct = completion.identity.prompt_idx == 7 or completion.identity.sample_idx > 0
        feedback = "correct" if correct else "wrong carry"
        return FTTTJudgeResult(
            correct,
            float(correct),
            feedback,
            (11, 12),
            "retry-contract",
            "scripted",
            "",
        )


def _gpu_shared_input() -> None:
    assert not torch.cuda.is_initialized()
    handle = pllm.make_llm(
        workload=ActivationWorkload(ActivationConfig((23,), 2, first_tokens=1, last_tokens=1)),
        config=engine(),
    )
    try:
        sampling = greedy(2)
        baseline = handle.generate(PROMPTS, sampling, feedback_group="cpu")
        assert not torch.cuda.is_initialized() and baseline.hidden.device.type == "cpu"
        width = baseline.hidden.shape[1]
        device = torch.device("cuda", 0)
        base = torch.zeros((width + 1, 2), device=device, dtype=torch.bfloat16)
        # A strided view with a storage offset must reach the child as the same values.
        factors = (
            base[1:].T,
            torch.ones(2, device=device, dtype=torch.bfloat16),
            torch.full((width, 2), 0.1, device=device, dtype=torch.bfloat16),
        )
        assert not factors[0].is_contiguous() and factors[0].storage_offset() > 0
        changed = handle.generate(
            PROMPTS,
            sampling,
            auxiliary=StateTensorUpdate(factors),
            feedback_group="gpu",
            completion_device=device,
        )
        assert changed.hidden.device == device and changed.readiness is not None
        changed.readiness.order_on(torch.cuda.current_stream(device))
        assert torch.isfinite(changed.hidden).all() and not torch.allclose(baseline.hidden, changed.hidden.cpu())
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.update_count == 1
        assert diagnostics.inference_pins == diagnostics.pending_state_releases == 0
        expected = changed.hidden.clone()
    finally:
        handle.close()
    # The CUDA completion is owned by the parent and survives the child.
    torch.testing.assert_close(changed.hidden, expected)
    print("EngineCore GPU IPC: strided CUDA input, publication=1, owned CUDA completion survives close")


def _user_reference_workload() -> None:
    """A user workload in an importable module enters EngineCore by reference, unchanged."""
    from examples.huber import HuberWorkload

    assert not torch.cuda.is_initialized()
    handle = pllm.make_llm(workload=HuberWorkload(2), config=engine())
    try:
        child = handle.worker_state().process_id
        assert child != os.getpid()
        assert len(handle.generate(PROMPTS, greedy(4))) == 2
        finished = handle.worker_state().diagnostics
        assert finished.loss_count == finished.update_count > 0
        assert_clean(finished)
        print(f"user workload by reference: child={child} loss={finished.loss_count}")
    finally:
        handle.close()


def _prefill_adapter_reconstruction() -> None:
    """A rebuilt worker starts fresh request identities and exact version-zero roots."""
    processes, requests = [], []
    for _ in range(2):
        with pllm.make_llm(workload=PrefillAdapters(chunk_size=3, rank=2), config=engine(memory=0.15)) as handle:
            handle.observe(TraceSelection(max_events=4096))
            outputs = handle.generate(["The quick brown fox jumps over the lazy dog.", "Hello there."], greedy(2))
            state = handle.worker_state()
            assert state.process_id != os.getpid()
            assert state.diagnostics.update_count == 5 * sum(
                (len(output.prompt_token_ids) + 2) // 3 for output in outputs
            )
            assert_clean(state.diagnostics)
            admissions = [event for event in handle.trace_snapshot().events if event.phase is TracePhase.ADMIT]
            assert admissions and all(ref.version_id == 0 for event in admissions for ref in event.refs)
            processes.append(state.process_id)
            requests.append({identity.request_id for event in admissions for identity in event.identities})
            assert not torch.cuda.is_initialized()
    assert processes[0] != processes[1] and requests[0].isdisjoint(requests[1])
    print(f"prefill adapters reconstructed workers={processes}; fresh requests={requests}; parent CUDA=off")


class ExportProbe(DummyWorkload):
    def on_export(self, event: pllm.TraceEvent) -> None:
        with open("build/worker_exports.pkl", "ab") as output:
            pickle.dump((event.sequence, event.step_id, event.tensor()), output)


def _dummy_export() -> None:
    from test.gpu.enginecore_smoke import ExportProbe

    assert not torch.cuda.is_initialized()
    Path("build").mkdir(exist_ok=True)
    Path("build/worker_exports.pkl").unlink(missing_ok=True)
    workload = ExportProbe(DummyConfig(layer=3, export_every_n_steps=2, export_max_rows=2, export_max_cols=3))
    handle = pllm.make_llm(workload=workload, config=engine())
    try:
        handle.generate(PROMPTS, greedy(4), prompt_indices=(7, 9))
        handle.generate(["The sea is"], greedy(3), prompt_indices=(11,))
        events = tuple(event for event in handle.trace_snapshot().events if event.phase is TracePhase.TENSOR)
        assert [event.step_id for event in events] == [2, 6]
        assert [tuple(identity.prompt_idx for identity in event.identities) for event in events] == [(7, 9), (11,)]
        for event in events:
            tensor = event.tensor()
            assert tensor.shape == (len(event.identities), 3) and tensor.dtype == torch.float32
            assert tensor.device.type == "cpu" and torch.isfinite(tensor).all() and tensor.abs().sum() > 0
        diagnostics = handle.worker_state().diagnostics
        assert_clean(diagnostics)
        assert diagnostics.loss_count == diagnostics.update_count == 0
        assert not torch.cuda.is_initialized()
        print("Dummy selected-layer export: snapshots=2 at the original rows and steps")
    finally:
        handle.close()

    with open("build/worker_exports.pkl", "rb") as source:
        for event in events:
            sequence, step, tensor = pickle.load(source)
            assert (sequence, step) == (event.sequence, event.step_id)
            torch.testing.assert_close(tensor, event.tensor())
        assert source.read() == b""


def _graph_overlay(*, exact: bool = False) -> None:
    """Bind multi-target pinned/hybrid overlays without initializing driver CUDA."""
    from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload

    assert not torch.cuda.is_initialized()
    targets = tuple(
        pllm.model.ParameterTarget(
            layer, pllm.model.ParameterModuleKind.ATTENTION, pllm.model.ParameterRole.QUERY_PROJECTION_WEIGHT
        )
        for layer in (0, 2)
    )
    workload = GatedDeltaWorkload(
        GatedDeltaConfig(targets, chunk_size=4, visibility=pllm.Exact() if exact else pllm.Visibility.forward_pinned())
    )
    options = ExecutionOptions(
        capture_memory_bytes=512 << 20, memory_budget_bytes=2 << 30, allow_numerical_transforms=False
    )
    with pllm.make_llm(workload=workload, config=engine(cuda_graphs=True, execution=options)) as handle:
        prompts = ["The sky is blue and the sea is calm.", "Water freezes when it is very cold."]
        first = handle.generate(prompts, greedy(3))
        second = handle.generate(prompts[::-1], greedy(3))
        assert [o.outputs[0].token_ids for o in first] == [o.outputs[0].token_ids for o in second][::-1]
        state = handle.worker_state()
        assert state.process_id != os.getpid() and state.diagnostics.update_count > 0
        assert_clean(state.diagnostics)
        assert not torch.cuda.is_initialized()
    print(f"Multi-target source graphs: exact={exact}; EngineCore replay; parent CUDA=off")


def _captured_esamp(runner: RunnerName = "mrv2") -> None:
    assert not torch.cuda.is_initialized()
    options = ExecutionOptions(capture_memory_bytes=16 * 1024**2)
    handle = pllm.make_llm(
        workload=ESampRequestWorkload(ESampRequestConfig(rank=2)),
        config=engine(runner=runner, execution=options),
    )
    try:
        tasks = handle.attachment.plan.schedule.tasks
        assert all("cuda_graph" in task.implementation_id for task in tasks)
        selected = tuple(task.task_id for task in tasks)
        handle.profile_tasks(selected)
        loss = next(task.task_id for task in tasks if ":loss@" in task.implementation_id)
        handle.observe(TraceSelection(tasks=(loss,), gradients=True, kernel_calls=1))
        for prompts in (
            ["The sky is", "The sea is", "The grass is"],
            ["Water freezes at"],
        ):
            handle.generate(prompts, greedy(3))
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.loss_count == diagnostics.update_count == 8
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == 0
        assert not torch.cuda.is_initialized()
        trace = handle.trace_snapshot()
        assert trace.pending_device_timings == 0 and {task for task, _ in trace.device_task_times()} == set(selected)
        assert all(count > 0 and elapsed > 0 for count, elapsed in trace.device_task_times().values())
        assert any(event.phase is TracePhase.KERNEL and event.device_duration_ns > 0 for event in trace.events)
        assert any(event.phase is TracePhase.GRADIENT and event.tensor().abs().sum() > 0 for event in trace.events)
        handle.observe(TraceSelection())
        handle.profile_tasks(())
        print(f"Captured ESamp EngineCore ({runner}): loss/update=8, profiled tasks, parent CUDA=off")
    finally:
        handle.close()


def _tlm_shared_input(model_id: str = MODEL) -> None:
    if not Path(model_id).is_dir():
        patterns = ("*.json", "*.safetensors", "*.txt", "*.model")
        model_id = snapshot_download(model_id, local_files_only=True, allow_patterns=patterns)
    assert not torch.cuda.is_initialized()
    handle = pllm.make_llm(workload=TLMWorkload(TLMConfig(threshold=0.0), rank=2, alpha=4), config=engine(model_id))
    try:
        tokens = torch.tensor([[1, 2, 3, 4]])
        batch = handle.generate(
            PROMPTS,
            greedy(2),
            auxiliary=CausalTrainBatch(tokens, tokens),
            feedback_group="tlm",
        )
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.loss_count == diagnostics.update_count == 1 and diagnostics.trained_rows > 0
        assert diagnostics.inference_pins == diagnostics.pending_state_releases == 0
        assert batch.hidden.device.type == "cpu" and torch.isfinite(batch.hidden).all()
        assert not torch.cuda.is_initialized()
        # Without a training batch the next invocation does not update.
        handle.generate("A second invocation", SamplingParams(max_tokens=2, temperature=0.0))
        assert handle.worker_state().diagnostics.update_count == 1
        print("TLM borrowed worker model: loss/update=1, parent CUDA=off")
    finally:
        handle.close()


def _output_lora_shared_input() -> None:
    assert not torch.cuda.is_initialized()
    workload = OutputLoRAWorkload(OutputLoRAConfig(rank=2, learning_rate=1.0, steps=2, seed=7))
    # This case asserts immediate next-draw visibility; exercise strict ordering.
    handle = pllm.make_llm(
        workload=workload,
        config=engine(execution=ExecutionOptions(training_rows=2, publish_lag=0)),
    )
    try:
        sampling = greedy(3, logprobs=1)
        baseline = handle.generate(PROMPTS, sampling)
        captured = handle.generate(PROMPTS, sampling, feedback_group="output-lora")
        batch = WeightedCEBatch(
            captured.hidden,
            torch.zeros(2, 151936),
            torch.tensor([entry.token_ids[-1] for entry in captured.completions.entries]),
            torch.tensor([1.0, 3.0]),
        )
        trained = handle.generate(PROMPTS, sampling, auxiliary=batch)
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.loss_count == 2 and diagnostics.update_count == 1 and diagnostics.trained_rows == 2
        assert diagnostics.inference_pins == diagnostics.pending_state_releases == 0
        changed = False
        for old, new in zip(baseline, trained, strict=True):
            old, new = old.outputs[0], new.outputs[0]
            assert old.token_ids[:2] == new.token_ids[:2], "training must publish after the first decode draw"
            old_logprob, new_logprob = (
                old.logprobs[2][old.token_ids[2]].logprob,
                new.logprobs[2][new.token_ids[2]].logprob,
            )
            changed |= old.token_ids[2] != new.token_ids[2] or abs(old_logprob - new_logprob) > 1e-5
        assert changed, "published weighted-CE factors never affected the actual sampler"
        handle.generate(PROMPTS, sampling)
        assert handle.worker_state().diagnostics.update_count == 1

        service = FTTTFeedbackService(FTTTConfig(rank=2, max_retries=3), RetryOnce())
        processor = FTTTWorkload(service, hidden_size=captured.hidden.shape[1])
        rounds = list(
            handle.generate_feedback(
                PROMPTS,
                sampling,
                feedback=processor.run,
                feedback_group="retry",
                max_rounds=3,
                prompt_indices=(7, 9),
            )
        )
        prompt_rounds = [tuple(entry.identity.prompt_idx for entry in batch.completions.entries) for batch, _ in rounds]
        assert prompt_rounds == [(7, 9), (9,)]
        assert sum(result.update_count for _, result in rounds) == 1
        assert service.instance_state("retry", 9).retries == 1 and service.instance_state("retry", 9).stopped
        assert service.queued_transactions == service.queued_feedback == 0
        assert all(batch.hidden.device.type == "cpu" for batch, _ in rounds)
        assert not torch.cuda.is_initialized()
        print(
            "Output LoRA EngineCore: shared publication=1, real sampler changed; feedback retry: 2 prompts -> 1 retry"
        )
    finally:
        handle.close()


def _independent_lora_input() -> None:
    assert not torch.cuda.is_initialized()
    workload = OutputLoRAWorkload(
        OutputLoRAConfig(rank=2, learning_rate=10.0, steps=2, seed=7, scope=StateScope.REQUEST)
    )
    handle = pllm.make_llm(workload=workload, config=engine(execution=ExecutionOptions()))
    try:
        # Default lag 1 makes the first decode update visible on the fourth draw.
        sampling = greedy(4, logprobs=1)
        baseline = handle.generate(PROMPTS, sampling)
        completed = handle.generate(PROMPTS, sampling, feedback_group="independent-lora")
        fields = (RequestKeyField.PROMPT_IDX, RequestKeyField.SAMPLE_IDX)
        keys = tuple(RequestStateKey.project(RequestIdentity("", index, 0), fields) for index in (7, 9))
        batch = WeightedCEBatch(
            completed.hidden.unsqueeze(1),
            torch.zeros(2, 1, 151936),
            torch.tensor([[entry.token_ids[-1]] for entry in completed.completions.entries]),
            torch.ones(2, 1),
            keys,
        )
        handle.observe(TraceSelection(tensors=True, gradients=True, max_tensor_bytes=64))
        trained = handle.generate(PROMPTS, sampling, auxiliary=batch, prompt_indices=(7, 9))
        events = handle.trace_snapshot().events
        gradients = [event for event in events if event.phase is TracePhase.GRADIENT]
        assert len(gradients) == 4 and {identity.prompt_idx for e in gradients for identity in e.identities} == {7, 9}
        assert all(event.refs and torch.isfinite(event.tensor()).all() for event in gradients)
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.loss_count == 4 and diagnostics.update_count == 2 and diagnostics.trained_rows == 2
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == 0
        # Each request's own published version guides its sampler.
        guided = [event for event in events if event.tensor_name == "guidance" and event.refs[0].version_id > 0]
        assert len(guided) == 2 and {event.row.identity.prompt_idx for event in guided} == {7, 9}
        assert all(torch.isfinite(event.tensor()).all() and torch.count_nonzero(event.tensor()) > 0 for event in guided)
        for before, after in zip(baseline, trained, strict=True):
            assert before.outputs[0].token_ids[:3] == after.outputs[0].token_ids[:3]
        assert not torch.cuda.is_initialized()
        print("Independent LoRA EngineCore: captured SGD, two request publications, nonzero sampler guidance")
    finally:
        handle.close()


def _recirculation() -> None:
    assert not torch.cuda.is_initialized()
    workload = RecirculationWorkload(destination=22, alpha=0.6, ramp_tokens=2)
    handle = pllm.make_llm(workload=workload, config=engine(tokens=64))
    try:
        first = handle.generate(PROMPTS, greedy(2))
        second = handle.generate(PROMPTS, greedy(2))
        tokens = lambda results: tuple(tuple(output.outputs[0].token_ids) for output in results)
        assert tokens(first) == tokens(second) and all(len(ids) == 2 for ids in tokens(first))
        replay = tuple(event for event in handle.trace_snapshot().events if event.phase is TracePhase.REPLAY)
        assert replay and {event.row.identity.prompt_idx for event in replay} == {
            0,
            1,
            2,
            3,
        }
        assert_clean(handle.worker_state().diagnostics)
        assert not torch.cuda.is_initialized()
        print(f"Recirculation EngineCore: serial replay rows={len(replay)}")
    finally:
        handle.close()


def _fttt_activation_retry() -> None:
    config = ActivationConfig(
        (23,),
        rank=1,
        first_tokens=0,
        last_tokens=1,
        operator=ActivationPatchOperator.ADD,
        scope=StateScope.REQUEST,
    )
    handle = pllm.make_llm(workload=ActivationWorkload(config), config=engine(seqs=1))
    try:
        service = FTTTFeedbackService(FTTTConfig(rank=2, learning_rate=0.1, max_retries=3), RetryOnce())
        processor = FTTTWorkload(service, hidden_size=896)
        rounds = list(
            handle.generate_feedback(
                ["The sky is", "Water freezes at", "The sea is"],
                greedy(1),
                feedback=processor.run,
                edit=FTTTResult.state_update,
                feedback_group="edited-retry",
                max_rounds=3,
                prompt_indices=(7, 9, 11),
            )
        )
        prompt_rounds = [tuple(entry.identity.prompt_idx for entry in batch.completions.entries) for batch, _ in rounds]
        assert prompt_rounds == [(7, 9, 11), (9, 11)]
        # Retried prompts run with their FTTT edit applied to the real model's hidden state.
        before, after = rounds[0][0].hidden, rounds[1][0].hidden
        assert not torch.equal(before[1], after[0]) and not torch.equal(before[2], after[1])
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.update_count == 2, "each queued retry must import its own edit once"
        assert_clean(diagnostics)
        assert not torch.cuda.is_initialized()
        print("FTTT activation retry: 3 prompts -> 2 edited retries, 2 page publications")
    finally:
        handle.close()


def main() -> int:
    assert not torch.cuda.is_initialized()
    workload = ESampRequestWorkload(ESampRequestConfig(rank=2, source_layer=0, target_layer=-1, intervention_beta=0.2))
    handle = pllm.make_llm(workload=workload, config=engine(memory=0.35))
    try:
        assert not torch.cuda.is_initialized(), "frontend allocated a second GPU execution"
        child = handle.worker_state().process_id
        assert child != os.getpid()
        outputs = handle.generate(
            ["The sky is", "The sea is", "The grass is"],
            greedy(3),
            prompt_indices=(9, 7, 11),
            sample_index=2,
        )
        assert len(outputs) == 3 and all(len(output.outputs[0].token_ids) == 3 for output in outputs)
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.loss_count == diagnostics.update_count == 6
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == 0
        again = handle.generate(["A second invocation"], greedy(3))
        assert len(again[0].outputs[0].token_ids) == 3
        final = handle.worker_state().diagnostics
        assert final.loss_count == final.update_count == 8
        assert final.active_lineages == final.pending_state_releases == 0
        trace = handle.trace_snapshot()
        assert trace.process_id == child and trace.host_task_times()
        assert all(event.phase is not TracePhase.TASK_FAILED for event in trace.events)
        # One shared completion batch feeds both host workloads.
        regex = StructuredOutputsParams(regex=r"\\boxed\{[12]\}")
        batch = handle.generate(
            ["Choose 1 or 2, boxed: "],
            SamplingParams(n=3, max_tokens=16, temperature=0.8, structured_outputs=regex),
            feedback_group="vote",
            prompt_indices=(100,),
            sample_index=2,
        )
        assert [(entry.identity.prompt_idx, entry.identity.sample_idx) for entry in batch.completions.entries] == [
            (100, 2),
            (100, 3),
            (100, 4),
        ]
        assert batch.hidden.device.type == "cpu" and batch.readiness is None
        assert batch.hidden.shape == (3, handle.attachment.probe.hidden_size)
        assert torch.isfinite(batch.hidden).all() and torch.count_nonzero(batch.hidden) > 0
        facts = HostConfig(torch.device("cpu"), hidden_size=batch.hidden.shape[1])
        feedback = FTTTFeedbackService(FTTTConfig(), FTTTDefaultJudge())
        edited = FTTTWorkload(feedback, hidden_size=facts.hidden_size).run(batch)
        assert edited is not None and edited.update_count == 1
        ttrl = TTRLConfig(rank=2, votes_per_prompt=3, samples_per_prompt=2)
        reward = make_host(facts, TTRLWorkload(ttrl))
        reward.admit(())
        trained = reward.step(ttrl_rewards(batch, ttrl).step)
        assert trained.loss is not None and reward.diagnostics().update_count == 1
        reward.retire(())
        assert not torch.cuda.is_initialized()
        print(f"enginecore child={child} loss={final.loss_count} update={final.update_count}")
    finally:
        handle.close()
    # Inputs use the original StateTensorUpdate; outputs outlive the child and exporting FD.
    reft = pllm.make_llm(
        workload=ActivationWorkload(ActivationConfig((23,), 2, first_tokens=1, last_tokens=1)),
        config=engine(memory=0.35),
    )
    try:
        baseline = reft.generate(PROMPTS, greedy(2), feedback_group="before")
        width = baseline.hidden.shape[1]
        parameters = StateTensorUpdate(
            (
                torch.zeros(2, width, dtype=torch.bfloat16),
                torch.ones(2, dtype=torch.bfloat16),
                torch.full((width, 2), 0.1, dtype=torch.bfloat16),
            )
        )
        changed = reft.generate(PROMPTS, greedy(2), auxiliary=parameters, feedback_group="after")
        assert not torch.equal(baseline.hidden, changed.hidden)
        diagnostics = reft.worker_state().diagnostics
        assert diagnostics.update_count == 1
        assert diagnostics.inference_pins == diagnostics.pending_state_releases == 0
        assert not torch.cuda.is_initialized()
    finally:
        reft.close()
    assert torch.isfinite(batch.hidden).all() and torch.isfinite(changed.hidden).all()
    print("shared completion -> FTTT/TTRL updates=1/1; shared ReFT input publications=1; parent CUDA=off")
    _tlm_shared_input()
    _output_lora_shared_input()
    _independent_lora_input()
    _fttt_activation_retry()
    _captured_esamp()
    _dummy_export()
    _user_reference_workload()
    _prefill_adapter_reconstruction()
    _graph_overlay()
    _graph_overlay(exact=True)
    _gpu_shared_input()
    return 0


MODES = {
    "--captured-only": _captured_esamp,
    "--pinned-graph-only": _graph_overlay,
    "--hybrid-graph-only": lambda: _graph_overlay(exact=True),
    "--mrv1-captured": lambda: _captured_esamp("mrv1"),
    "--gpu-io-only": _gpu_shared_input,
    "--lora-only": lambda: (_output_lora_shared_input(), _independent_lora_input()),
    "--edit-only": _fttt_activation_retry,
    "--recirculation-only": _recirculation,
    "--dummy-only": _dummy_export,
    "--user-only": _user_reference_workload,
    "--prefill-adapters-only": _prefill_adapter_reconstruction,
    "--tlm-only": _tlm_shared_input,
}

if __name__ == "__main__":
    if sys.argv[1:]:
        MODES[sys.argv[1]]()
    else:
        raise SystemExit(main())

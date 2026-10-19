"""Evidence 36: original signed ESamp execution in a real EngineCore child."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from huggingface_hub import snapshot_download
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

import pllm
from pllm.authoring import CausalTrainBatch, StateTensorUpdate, WeightedCEBatch
from pllm.backends.vllm.runtime import BackendCommand, VllmLlm
from pllm.config import ExecutionOptions, ModelConfig, RunnerKind, VllmDType
from pllm.contracts.backend_attachment import ActivationPatchOperator, AttachmentTopology
from pllm.contracts.region_imports import (
    BorrowedTensorImport,
    ExportedTensorImport,
    FunctionalKernelImport,
    NativeKernelImport,
)
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.semantic_graph import StateScope
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.host_runtime import HostRuntimeFacts
from pllm.execution.trace import TracePhase, TraceSelection
from pllm.workloads.dummy import DummyConfig, DummyWorkload
from pllm.workloads.esamp_request import ESampRequestConfig, ESampRequestWorkload
from pllm.workloads.fttt import FTTTConfig, FTTTFeedbackService, FTTTJudgeResult, FTTTJudgeService, FTTTWorkload
from pllm.workloads.fttt.types import FTTTResult
from pllm.workloads.lora import OutputLoRAConfig, OutputLoRAWorkload
from pllm.workloads.recirculation import RecirculationConfig, RecirculationWorkload
from pllm.workloads.reft import ActivationConfig, ActivationWorkload
from pllm.workloads.tlm import HfQvConfig, HfQvModel, TLMConfig, TLMWorkload
from pllm.workloads.ttrl import TTRLConfig, TTRLWorkload


def _gpu_shared_input() -> None:
    assert not torch.cuda.is_initialized()
    handle = pllm.make_llm(
        workload=ActivationWorkload(ActivationConfig((23,), 2, first_tokens=1, last_tokens=1)),
        config=ModelConfig(
            "Qwen/Qwen2.5-0.5B-Instruct", VllmDType.BFLOAT16, 64, 2, 128, 0.25, seed=7, multiprocessing=True
        ),
    )
    try:
        prompts = ["The sky is", "Water freezes at"]
        sampling = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
        baseline = handle.generate(prompts, sampling, feedback_group="cpu")
        assert not torch.cuda.is_initialized() and baseline.hidden.device.type == "cpu"
        width = baseline.hidden.shape[1]
        device = torch.device("cuda", 0)
        base = torch.zeros((width + 1, 2), device=device, dtype=torch.bfloat16)
        factors = (
            base[1:].T,
            torch.ones(2, device=device, dtype=torch.bfloat16),
            torch.zeros((width, 2), device=device, dtype=torch.bfloat16),
        )
        assert not factors[0].is_contiguous() and factors[0].storage_offset() > 0
        original_command = VllmLlm.command

        def staged_command(self, command, payload=""):
            result = original_command(self, command, payload)
            if command is BackendCommand.STAGE:
                # Scheduling is still paused: this proves the imported view aliases
                # the source allocation, rather than a hidden transport-time clone.
                factors[2].fill_(0.1)
                torch.cuda.current_stream(device).synchronize()
            return result

        with patch.object(VllmLlm, "command", staged_command):
            changed = handle.generate(
                prompts, sampling, auxiliary=StateTensorUpdate(factors), feedback_group="gpu", completion_device=device
            )
        assert changed.hidden.device == device and changed.readiness is not None
        changed.readiness.order_on(torch.cuda.current_stream(device))
        assert torch.isfinite(changed.hidden).all()
        assert not torch.allclose(baseline.hidden, changed.hidden.cpu())
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.update_count == 1
        assert diagnostics.inference_pins == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        expected = changed.hidden.clone()
    finally:
        handle.close()
    torch.testing.assert_close(changed.hidden, expected)
    print(
        "EngineCore GPU IPC: original strided input alias, publication=1, owned CUDA completion survives close, fallback=0"
    )


def _user_reference_workload() -> None:
    """A user workload in an importable module enters EngineCore by reference, unchanged."""
    from examples.huber import HuberWorkload

    assert not torch.cuda.is_initialized()
    handle = pllm.make_llm(
        workload=HuberWorkload(2),
        config=ModelConfig(
            "Qwen/Qwen2.5-0.5B-Instruct", VllmDType.BFLOAT16, 64, 2, 128, 0.25, seed=7, multiprocessing=True
        ),
    )
    try:
        initial = handle.worker_state()
        assert initial.process_id != os.getpid()
        outputs = handle.generate(
            ["The sky is", "Water freezes at"], SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True)
        )
        assert len(outputs) == 2
        finished = handle.worker_state().diagnostics
        assert finished.loss_count == finished.update_count > 0
        assert finished.active_lineages == finished.pending_state_releases == finished.fallback_count == 0
        print(
            f"user workload by reference: child={initial.process_id} "
            f"loss={finished.loss_count} fallback={finished.fallback_count}"
        )
    finally:
        handle.close()


def _dummy_export(multiprocessing: bool = True) -> None:
    if multiprocessing:
        assert not torch.cuda.is_initialized()
    handle = pllm.make_llm(
        workload=DummyWorkload(DummyConfig(layer=3, export_every_n_steps=2, export_max_rows=2, export_max_cols=3)),
        config=ModelConfig(
            "Qwen/Qwen2.5-0.5B-Instruct", VllmDType.BFLOAT16, 64, 2, 128, 0.25, seed=7, multiprocessing=multiprocessing
        ),
    )
    expected = []
    hook = None
    if not multiprocessing:

        def observe(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            expected.append(hidden.detach()[:, :3].to(device="cpu", dtype=torch.float32, copy=True))

        hook = handle.runner.model.get_submodule("model.layers.3").register_forward_hook(observe)
    try:
        handle.generate(
            ["The sky is", "Water freezes at"],
            SamplingParams(max_tokens=4, temperature=0.0, ignore_eos=True),
            prompt_indices=(7, 9),
        )
        handle.generate(
            ["The sea is"], SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True), prompt_indices=(11,)
        )
        events = tuple(event for event in handle.trace_snapshot().events if event.phase is TracePhase.TENSOR)
        assert len(events) == 2 and [event.step_id for event in events] == [2, 6]
        assert [tuple(identity.prompt_idx for identity in event.identities) for event in events] == [(7, 9), (11,)]
        for event in events:
            tensor = event.tensor()
            assert tensor.shape == (len(event.identities), 3) and tensor.dtype == torch.float32
            assert tensor.device.type == "cpu" and torch.isfinite(tensor).all() and tensor.abs().sum() > 0
            if not multiprocessing:
                torch.testing.assert_close(tensor, expected[event.step_id], rtol=0, atol=0)
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.active_lineages == diagnostics.inference_pins == diagnostics.pending_state_releases == 0
        assert diagnostics.loss_count == diagnostics.update_count == diagnostics.fallback_count == 0
        if multiprocessing:
            assert not torch.cuda.is_initialized()
        print(
            f"Dummy selected-layer export: snapshots=2, original rows/steps, independent checks={len(events) if expected else 0}, MP={multiprocessing}, fallback=0"
        )
    finally:
        if hook is not None:
            hook.remove()
        handle.close()


def _captured_esamp(runner: RunnerKind = RunnerKind.MRV2) -> None:
    assert not torch.cuda.is_initialized()
    handle = pllm.make_llm(
        workload=ESampRequestWorkload(ESampRequestConfig(rank=2)),
        config=ModelConfig(
            "Qwen/Qwen2.5-0.5B-Instruct",
            VllmDType.BFLOAT16,
            64,
            2,
            128,
            0.25,
            seed=7,
            multiprocessing=True,
            runner=runner,
            execution=ExecutionOptions(capture=True, capture_memory_bytes=16 * 1024**2),
        ),
    )
    try:
        tasks = handle.attachment.inputs.plan.schedule_physical.tasks
        selected = tuple(task.task_id for task in tasks if "cuda_graph" in task.implementation_id)
        assert len(selected) == len(tasks)
        handle.profile_tasks(selected)
        loss = next(task.task_id for task in tasks if ":loss@" in task.implementation_id)
        handle.observe(TraceSelection(tasks=(loss,), gradients=True, kernel_calls=1))
        for prompts in (["The sky is", "The sea is", "The grass is"], ["Water freezes at"]):
            handle.generate(prompts, SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True))
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.loss_count == diagnostics.update_count == 8
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        assert not torch.cuda.is_initialized()
        trace = handle.trace_snapshot()
        assert trace.schema_version == 5 and trace.pending_device_timings == 0
        assert {task for task, _ in trace.device_task_times()} == set(selected)
        assert all(count > 0 and elapsed > 0 for count, elapsed in trace.device_task_times().values())
        assert any(event.phase is TracePhase.KERNEL and event.device_duration_ns > 0 for event in trace.events)
        assert any(event.phase is TracePhase.GRADIENT and event.tensor().abs().sum() > 0 for event in trace.events)
        handle.observe(TraceSelection())
        handle.profile_tasks(())
        print(
            f"Captured ESamp EngineCore ({runner}): loss/update=8, dynamic rows within signed capacity, parent CUDA=off, fallback=0"
        )
    finally:
        handle.close()


def _tlm_shared_input(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
    model_id = (
        model_id
        if Path(model_id).is_dir()
        else snapshot_download(
            model_id, local_files_only=True, allow_patterns=("*.json", "*.safetensors", "*.txt", "*.model")
        )
    )
    model = HfQvModel.load(
        HfQvConfig(model_id, "main", True, False, 2, 4, 0.0, TensorDType.BFLOAT16), torch.device("meta")
    )
    assert all(factor.device.type == "meta" for factor in model.initial_factors)
    assert not torch.cuda.is_initialized()
    handle = pllm.make_llm(
        workload=TLMWorkload(TLMConfig(threshold=0.0), model),
        config=ModelConfig(model_id, VllmDType.BFLOAT16, 64, 2, 128, 0.25, seed=7, multiprocessing=True),
    )
    assert model.resident_bytes > 0 and all(
        value >= model.resident_bytes for value in handle.attachment.inputs.plan.timeline.reserved_bytes
    )
    try:
        tokens = torch.tensor([[1, 2, 3, 4]])
        batch = handle.generate(
            ["The sky is", "Water freezes at"],
            SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True),
            auxiliary=CausalTrainBatch(tokens, tokens),
            feedback_group="tlm",
        )
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.loss_count == diagnostics.update_count == 1 and diagnostics.trained_rows > 0
        assert diagnostics.inference_pins == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        assert batch.hidden.device.type == "cpu" and torch.isfinite(batch.hidden).all()
        assert not torch.cuda.is_initialized()
        handle.generate("A second invocation", SamplingParams(max_tokens=2, temperature=0.0))
        assert handle.worker_state().diagnostics.update_count == 1
        print(
            f"TLM closed model reconstruction: loss/update=1, shared causal input, parent CUDA=off, model bytes={model.resident_bytes}"
        )
    finally:
        handle.close()
    assert handle._llm is handle._runner is handle._deployment is None
    assert torch.isfinite(batch.hidden).all()


def _output_lora_shared_input() -> None:
    assert not torch.cuda.is_initialized()
    handle = pllm.make_llm(
        workload=OutputLoRAWorkload(OutputLoRAConfig(rank=2, learning_rate=1.0, steps=2, seed=7)),
        config=ModelConfig(
            "Qwen/Qwen2.5-0.5B-Instruct",
            VllmDType.BFLOAT16,
            64,
            2,
            128,
            0.25,
            seed=7,
            multiprocessing=True,
            execution=ExecutionOptions(training_rows=2),
        ),
    )
    try:
        prompts = ["The sky is", "Water freezes at"]
        sampling = SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True, logprobs=1)
        baseline = handle.generate(prompts, sampling)
        captured = handle.generate(prompts, sampling, feedback_group="output-lora")
        batch = WeightedCEBatch(
            captured.hidden,
            torch.zeros(2, 151936),
            torch.tensor([entry.token_ids[-1] for entry in captured.completions.entries]),
            torch.tensor([1.0, 3.0]),
        )
        trained = handle.generate(prompts, sampling, auxiliary=batch)
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.loss_count == 2 and diagnostics.update_count == 1 and diagnostics.trained_rows == 2
        assert diagnostics.inference_pins == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        changed = False
        for old, new in zip(baseline, trained, strict=True):
            old, new = old.outputs[0], new.outputs[0]
            assert old.token_ids[:2] == new.token_ids[:2], "training must publish after the first decode draw"
            changed |= (
                old.token_ids[2] != new.token_ids[2]
                or abs(old.logprobs[2][old.token_ids[2]].logprob - new.logprobs[2][new.token_ids[2]].logprob) > 1e-5
            )
        assert changed, "published weighted-CE factors never affected the actual sampler"
        handle.generate(prompts, sampling)
        assert handle.worker_state().diagnostics.update_count == 1

        class RetryOnce(FTTTJudgeService):
            def judge(self, completion):
                correct = completion.identity.prompt_idx == 7 or completion.identity.sample_idx > 0
                return FTTTJudgeResult(
                    correct,
                    float(correct),
                    "correct" if correct else "wrong carry",
                    (11, 12),
                    "retry-contract",
                    "scripted",
                    "",
                )

        service = FTTTFeedbackService(FTTTConfig(rank=2, max_retries=3), RetryOnce())
        processor = pllm.make_llm(
            workload=FTTTWorkload(service),
            config=HostRuntimeFacts(torch.device("cpu"), hidden_size=captured.hidden.shape[1]),
        )
        rounds = list(
            handle.generate_feedback(
                prompts, sampling, feedback=processor.run, feedback_group="retry", max_rounds=3, prompt_indices=(7, 9)
            )
        )
        assert [tuple(entry.identity.prompt_idx for entry in batch.completions.entries) for batch, _ in rounds] == [
            (7, 9),
            (9,),
        ]
        assert sum(result.update_count for _, result in rounds) == 1
        assert service.instance_state("retry", 9).retries == 1 and service.instance_state("retry", 9).stopped
        assert service.queued_transactions == service.queued_feedback == 0
        assert all(batch.hidden.device.type == "cpu" for batch, _ in rounds)
        assert not torch.cuda.is_initialized()
        print(
            "Output LoRA EngineCore: SGD steps=2, shared publication=1, trained rows=2, real sampler changed, parent CUDA=off"
        )
        print(
            "Automatic feedback retry: two original prompts -> one retry, one settled FTTT update, then accept; edits not applied"
        )
    finally:
        handle.close()


def _independent_lora_input() -> None:
    assert not torch.cuda.is_initialized()
    handle = pllm.make_llm(
        workload=OutputLoRAWorkload(
            OutputLoRAConfig(rank=2, learning_rate=10.0, steps=2, seed=7, scope=StateScope.REQUEST)
        ),
        config=ModelConfig(
            "Qwen/Qwen2.5-0.5B-Instruct",
            VllmDType.BFLOAT16,
            64,
            2,
            128,
            0.25,
            seed=7,
            multiprocessing=True,
            execution=ExecutionOptions(capture=True),
        ),
    )
    try:
        prompts = ["The sky is", "Water freezes at"]
        sampling = SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True, logprobs=1)
        baseline = handle.generate(prompts, sampling)
        completed = handle.generate(prompts, sampling, feedback_group="independent-lora")
        keys = tuple(
            RequestStateKey.project(
                RequestIdentity("", index, 0), (RequestKeyField.PROMPT_IDX, RequestKeyField.SAMPLE_IDX)
            )
            for index in (7, 9)
        )
        batch = WeightedCEBatch(
            completed.hidden.unsqueeze(1),
            torch.zeros(2, 1, 151936),
            torch.tensor([[entry.token_ids[-1]] for entry in completed.completions.entries]),
            torch.ones(2, 1),
            keys,
        )
        handle.observe(TraceSelection(tensors=True, gradients=True, max_tensor_bytes=64))
        trained = handle.generate(prompts, sampling, auxiliary=batch, prompt_indices=(7, 9))
        gradients = [event for event in handle.trace_snapshot().events if event.phase is TracePhase.GRADIENT]
        assert len(gradients) == 4 and {
            identity.prompt_idx for event in gradients for identity in event.identities
        } == {7, 9}
        assert all(event.refs and torch.isfinite(event.tensor()).all() for event in gradients)
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.loss_count == 4 and diagnostics.update_count == 2 and diagnostics.trained_rows == 2
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        guided = [
            event
            for event in handle.trace_snapshot().events
            if event.tensor_name == "guidance" and event.refs[0].version_id > 0
        ]
        assert len(guided) == 2 and {event.row.identity.prompt_idx for event in guided} == {7, 9}
        assert all(torch.isfinite(event.tensor()).all() and torch.count_nonzero(event.tensor()) > 0 for event in guided)
        for before, after in zip(baseline, trained, strict=True):
            old, new = before.outputs[0], after.outputs[0]
            assert old.token_ids[:2] == new.token_ids[:2]
        assert not torch.cuda.is_initialized()
        print(
            "Independent LoRA EngineCore: captured SGD, loss=4, two exact request publications, nonzero signed sampler guidance, cleanup/fallback=0"
        )
    finally:
        handle.close()


def _recirculation_schedules(measured: bool = False) -> None:
    assert not torch.cuda.is_initialized()
    original_command = VllmLlm.command

    def command(self, kind, payload="", **kwargs):
        assert measured or not kind.value.startswith("tune_")
        return original_command(self, kind, payload, **kwargs)

    with patch.object(VllmLlm, "command", command):
        handle = pllm.make_llm(
            workload=RecirculationWorkload(
                RecirculationConfig(source_layer=-1, destination_layer=22, alpha=0.6, beta=None, ramp_tokens=2)
            ),
            config=ModelConfig(
                "Qwen/Qwen2.5-0.5B-Instruct",
                VllmDType.BFLOAT16,
                64,
                2,
                64,
                0.25,
                seed=7,
                multiprocessing=True,
                execution=ExecutionOptions(tuning="measured" if measured else "heuristic"),
            ),
        )
    try:
        selection = handle.region_selection
        assert (selection is not None) == measured
        imports = handle.attachment.inputs.probe.region_imports
        assert any(isinstance(item, ExportedTensorImport) for item in imports)
        assert any(isinstance(item, BorrowedTensorImport) for item in imports)
        assert any(
            isinstance(item, FunctionalKernelImport) and item.implementation.startswith("vllm.export-kernel@1:")
            for item in imports
        )
        assert len(handle.attachment.inputs.probe.region_bodies) == 2
        parallel_candidates, transformed_candidates = (), ()
        if measured:
            assert selection is not None and 1 < len(selection.candidates) <= 8
            assert all(candidate.feasible and candidate.median_ns is not None for candidate in selection.candidates)
            parallel_candidates = tuple(
                index
                for index, candidate in enumerate(selection.candidates)
                if any(
                    limit.bytes
                    > next(
                        device.peak_bytes_upper_bound
                        for device in candidate.payload.devices
                        if device.device == limit.device
                    )
                    for limit in candidate.peak_bytes
                )
            )
            assert parallel_candidates, "the model catalog must carry funded concurrent scratch through EngineCore"
            transformed_candidates = tuple(
                index for index, candidate in enumerate(selection.candidates) if candidate.changed_functions
            )
            assert transformed_candidates, "the worker must measure source-derived tensor implementations"
        assert handle.trace_snapshot().events == ()
        prompts = ["The sky is", "Water freezes at"]
        sampling = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
        first = handle.generate(prompts, sampling)
        second = handle.generate(prompts, sampling)
        tokens = lambda results: tuple(tuple(output.outputs[0].token_ids) for output in results)
        assert tokens(first) == tokens(second) and all(len(ids) == 2 for ids in tokens(first))
        replay = tuple(event for event in handle.trace_snapshot().events if event.phase is TracePhase.REPLAY)
        assert replay and {event.row.identity.prompt_idx for event in replay} == {0, 1, 2, 3}
        diagnostics = handle.worker_state().diagnostics
        assert (
            diagnostics.active_lineages
            == diagnostics.inference_pins
            == diagnostics.pending_state_releases
            == diagnostics.fallback_count
            == 0
        )
        assert not torch.cuda.is_initialized()
        print(
            dict(
                engine_core=True,
                mode="measured" if measured else "static",
                measured_ns=()
                if selection is None
                else tuple(candidate.median_ns for candidate in selection.candidates),
                parallel_candidates=parallel_candidates,
                transformed_candidates=transformed_candidates,
                kernel_imports=tuple(
                    kernel.implementation.split(":")[0]
                    for kernel in handle.attachment.inputs.probe.region_imports
                    if isinstance(kernel, (NativeKernelImport, FunctionalKernelImport))
                ),
                source_bodies=len(handle.attachment.inputs.probe.region_bodies),
                replay_rows=len(replay),
                parent_cuda=False,
                fallback=0,
            )
        )
    finally:
        handle.close()


def _fttt_activation_retry() -> None:
    handle = pllm.make_llm(
        workload=ActivationWorkload(
            ActivationConfig(
                (23,),
                rank=1,
                first_tokens=0,
                last_tokens=1,
                operator=ActivationPatchOperator.ADD,
                scope=StateScope.REQUEST,
            )
        ),
        config=ModelConfig(
            "Qwen/Qwen2.5-0.5B-Instruct", VllmDType.BFLOAT16, 64, 1, 128, 0.25, seed=7, multiprocessing=True
        ),
    )
    try:

        class RetryOnce(FTTTJudgeService):
            def judge(self, completion):
                correct = completion.identity.prompt_idx == 7 or completion.identity.sample_idx > 0
                return FTTTJudgeResult(
                    correct,
                    float(correct),
                    "correct" if correct else "wrong carry",
                    (11, 12),
                    "retry-contract",
                    "scripted",
                    "",
                )

        processor = pllm.make_llm(
            workload=FTTTWorkload(
                FTTTFeedbackService(FTTTConfig(rank=2, learning_rate=0.1, max_retries=3), RetryOnce())
            ),
            config=HostRuntimeFacts(torch.device("cpu"), hidden_size=896),
        )
        rounds = list(
            handle.generate_feedback(
                ["The sky is", "Water freezes at", "The sea is"],
                SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True),
                feedback=processor.run,
                edit=FTTTResult.state_update,
                feedback_group="edited-retry",
                max_rounds=3,
                prompt_indices=(7, 9, 11),
            )
        )
        assert [tuple(entry.identity.prompt_idx for entry in batch.completions.entries) for batch, _ in rounds] == [
            (7, 9, 11),
            (9, 11),
        ]
        before, after = rounds[0][0].hidden, rounds[1][0].hidden
        assert not torch.equal(before[1], after[0]) and not torch.equal(before[2], after[1])
        update = rounds[0][1].state_update()
        assert update.values[0] is rounds[0][1].edit_delta and len(update.keys) == 2
        diagnostics = handle.worker_state().diagnostics
        assert diagnostics.update_count == 2, "each queued retry must import its own edit once"
        assert (
            diagnostics.active_lineages
            == diagnostics.inference_pins
            == diagnostics.pending_state_releases
            == diagnostics.fallback_count
            == 0
        )
        assert not torch.cuda.is_initialized()
        print(
            "FTTT activation retry: 3 prompts -> 2 edited retries, 2 original page publications, changed real model hidden; MP=True"
        )
    finally:
        handle.close()


def main() -> int:
    assert not torch.cuda.is_initialized()
    handle = pllm.make_llm(
        workload=ESampRequestWorkload(
            ESampRequestConfig(rank=2, source_layer=0, target_layer=-1, intervention=True, intervention_beta=0.2)
        ),
        config=ModelConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype=VllmDType.BFLOAT16,
            max_model_len=64,
            max_num_seqs=2,
            max_num_batched_tokens=128,
            gpu_memory_utilization=0.35,
            seed=7,
            multiprocessing=True,
        ),
    )
    try:
        assert not torch.cuda.is_initialized(), "frontend allocated a second GPU execution"
        assert handle.attachment.inputs.probe.topology is AttachmentTopology.ENGINE_CORE
        initial = handle.worker_state()
        assert initial.process_id != os.getpid()
        assert initial.diagnostics.active_lineages == 0
        outputs = handle.generate(
            ["The sky is", "The sea is", "The grass is"],
            SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True),
            prompt_indices=(9, 7, 11),
            sample_index=2,
        )
        assert len(outputs) == 3 and all(len(output.outputs[0].token_ids) == 3 for output in outputs)
        finished = handle.worker_state()
        assert finished.process_id == initial.process_id
        diagnostics = finished.diagnostics
        assert diagnostics.loss_count == diagnostics.update_count == 6
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        again = handle.generate(["A second invocation"], SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True))
        assert len(again[0].outputs[0].token_ids) == 3
        final = handle.worker_state().diagnostics
        assert final.loss_count == final.update_count == 8
        assert final.active_lineages == final.pending_state_releases == final.fallback_count == 0
        trace = handle.trace_snapshot()
        assert trace.process_id == initial.process_id
        assert trace.host_task_times() and trace.events[-1].phase is TracePhase.RETIRE
        begins = tuple(event for event in trace.events if event.phase is TracePhase.BEGIN)
        assert begins and all(event.pin_id is not None for event in begins)
        assert all(event.phase is not TracePhase.TASK_FAILED for event in trace.events)
        batch = handle.generate(
            ["Choose 1 or 2, boxed: "] * 3,
            SamplingParams(
                max_tokens=16, temperature=0.8, structured_outputs=StructuredOutputsParams(regex=r"\\boxed\{[12]\}")
            ),
            feedback_group="vote",
            prompt_indices=(100, 100, 100),
            sample_index=2,
        )
        assert batch.hidden.device.type == "cpu" and batch.readiness is None
        assert batch.hidden.shape == (3, handle.attachment.inputs.probe.hidden_size)
        assert torch.isfinite(batch.hidden).all() and torch.count_nonzero(batch.hidden) > 0
        facts = HostRuntimeFacts(torch.device("cpu"), hidden_size=batch.hidden.shape[1])
        feedback = pllm.make_llm(workload=FTTTWorkload(), config=facts)
        edited = feedback.run(batch)
        assert edited is not None and edited.update_count == 1
        reward = pllm.make_llm(
            workload=TTRLWorkload(TTRLConfig(rank=2, votes_per_prompt=3, samples_per_prompt=2)), config=facts
        )
        reward.admit(())
        try:
            trained = reward.step(batch)
            assert trained.loss is not None and reward.diagnostics().update_count == 1
            for original, judged, voted in zip(
                batch.completions.entries, edited.intent.reward_outcomes, trained.reward_outcomes, strict=True
            ):
                assert judged.completion is voted.completion is original
        finally:
            reward.retire(())
        assert not torch.cuda.is_initialized()
        print(
            f"enginecore child={initial.process_id} loss={final.loss_count} update={final.update_count} "
            f"active={final.active_lineages} pending={final.pending_state_releases} fallback={final.fallback_count}"
        )
    finally:
        handle.close()
    # Inputs use the original StateTensorUpdate; outputs outlive the child and exporting FD.
    reft = pllm.make_llm(
        workload=ActivationWorkload(ActivationConfig((23,), 2, first_tokens=1, last_tokens=1)),
        config=ModelConfig(
            model="Qwen/Qwen2.5-0.5B-Instruct",
            dtype=VllmDType.BFLOAT16,
            max_model_len=64,
            max_num_seqs=2,
            max_num_batched_tokens=128,
            gpu_memory_utilization=0.35,
            seed=7,
            multiprocessing=True,
        ),
    )
    try:
        sampling = SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
        baseline = reft.generate(["The sky is", "Water freezes at"], sampling, feedback_group="before")
        width = baseline.hidden.shape[1]
        parameters = StateTensorUpdate(
            (
                torch.zeros(2, width, dtype=torch.bfloat16),
                torch.ones(2, dtype=torch.bfloat16),
                torch.full((width, 2), 0.1, dtype=torch.bfloat16),
            )
        )
        changed = reft.generate(
            ["The sky is", "Water freezes at"], sampling, auxiliary=parameters, feedback_group="after"
        )
        assert not torch.equal(baseline.hidden, changed.hidden)
        diagnostics = reft.worker_state().diagnostics
        assert diagnostics.update_count == 1
        assert diagnostics.inference_pins == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
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
    _gpu_shared_input()
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--captured-only"]:
        _captured_esamp()
    elif sys.argv[1:] == ["--mrv1-captured"]:
        _captured_esamp(RunnerKind.MRV1)
    elif sys.argv[1:] == ["--gpu-io-only"]:
        _gpu_shared_input()
    elif sys.argv[1:] == ["--lora-only"]:
        _output_lora_shared_input()
        _independent_lora_input()
    elif sys.argv[1:] == ["--edit-only"]:
        _fttt_activation_retry()
    elif sys.argv[1:] == ["--recirculation-only"]:
        _recirculation_schedules()
    elif sys.argv[1:] == ["--recirculation-measured"]:
        _recirculation_schedules(measured=True)
    elif sys.argv[1:] == ["--dummy-only"]:
        _dummy_export()
    elif sys.argv[1:] == ["--user-only"]:
        _user_reference_workload()
    elif sys.argv[1:] == ["--tlm-only"]:
        _tlm_shared_input()
    else:
        raise SystemExit(main())

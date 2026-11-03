"""CPU explanation contracts: built-in coverage, cost attribution and budget fallback without CUDA or weights."""

import json
from dataclasses import asdict, replace

import pytest
import torch

import pllm
from pllm.workloads import (
    DoLaWorkload,
    DummyWorkload,
    ESampConfig,
    ESampWorkload,
    FTTTWorkload,
    InPlaceTTTConfig,
    InPlaceTTTWorkload,
    JacobiRecirculationWorkload,
    OutputLoRAWorkload,
    OverRIDEConfig,
    OverRIDEWorkload,
    PrefillAdaptersWorkload,
    PromptFactorsWorkload,
    RecirculationWorkload,
    ReFTConfig,
    ReFTWorkload,
    StepLoopWorkload,
    TLMConfig,
    TLMWorkload,
    TTRLWorkload,
)
from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload
from pllm.workloads.video import VideoWorkload


@pytest.fixture
def model_config(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "hidden_size": 8,
                "num_hidden_layers": 10,
                "vocab_size": 32,
                "intermediate_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
            }
        )
    )

    def forbidden(*args, **kwargs):
        pytest.fail("explain must not initialize CUDA, query GPU memory or open model weights")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "mem_get_info", forbidden)
    monkeypatch.setattr("safetensors.safe_open", forbidden)
    return pllm.ModelConfig(str(tmp_path), max_num_seqs=3, max_num_batched_tokens=16)


@pytest.mark.parametrize(
    "workload",
    (
        DummyWorkload(),
        DoLaWorkload(layers=(0, 2, 4)),
        ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5)),
        PromptFactorsWorkload(),
        OutputLoRAWorkload(),
        OverRIDEWorkload(OverRIDEConfig(rank=2, num_iterations=3)),
        PrefillAdaptersWorkload(rank=2, chunk_size=4),
        GatedDeltaWorkload(GatedDeltaConfig((pllm.model.Projection("down_proj", 0),), head_width=4, chunk_size=4)),
        InPlaceTTTWorkload(InPlaceTTTConfig((pllm.model.Projection("down_proj", 0),), 4, 0.01)),
        ReFTWorkload(ReFTConfig((0,))),
        RecirculationWorkload(),
        JacobiRecirculationWorkload(),
        StepLoopWorkload(),
        FTTTWorkload(),
        TTRLWorkload(),
        TLMWorkload(TLMConfig()),
        VideoWorkload(),
    ),
    ids=lambda workload: type(workload).__name__,
)
def test_builtin_explanations_are_cpu_only_and_json_ready(workload, model_config):
    report = pllm.explain(workload, model_config)
    assert json.loads(json.dumps(asdict(report)))["model"]["hidden_size"] == 8
    assert report.peak_memory_bytes >= sum(state.reserved_bytes for state in report.states)
    assert report.members and not report.budget_errors
    assert "[exact]" in str(report)
    if isinstance(workload, (StepLoopWorkload, PrefillAdaptersWorkload)):
        tokens = tuple(item for item in report.escape_hatches if item.setting == "tokens")
        assert tokens and all(item.is_default for item in tokens)
    if report.eager_reason and "exact train_in_forward" not in report.eager_reason:
        with pytest.raises(ValueError, match='cuda_graphs="auto"'):
            pllm.explain(workload, replace(model_config, cuda_graphs=True))


def test_esamp_attribution_and_budget_fallback(model_config):
    workload = ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5))
    report = pllm.explain(workload, model_config)
    members = {member.name: member for member in report.members}
    assert members["loss"].effect == "train"
    assert any(task.lane == "side" and task.captured for task in members["loss"].tasks)
    prediction = next(task for task in members["predict"].tasks if task.name == "predict")
    assert prediction.reads == ("distiller",)
    assert report.inline_peak_memory_bytes < report.overlap_threshold_bytes == report.peak_memory_bytes
    inline = pllm.explain(
        workload,
        replace(
            model_config,
            execution=pllm.ExecutionOptions(
                memory_budget_bytes=report.inline_peak_memory_bytes,
            ),
        ),
    )
    assert not inline.budget_errors and all(task.lane == "main" for member in inline.members for task in member.tasks)
    failed = pllm.explain(workload, replace(model_config, execution=pllm.ExecutionOptions(memory_budget_bytes=1)))
    assert failed.budget_errors and "max_num_batched_tokens" in str(failed)
    assert failed.peak_memory_bytes == report.peak_memory_bytes
    state_failed = pllm.explain(workload, replace(model_config, execution=pllm.ExecutionOptions(state_memory_bytes=1)))
    assert state_failed.budget_errors


def test_declared_flop_estimates(model_config):
    report = pllm.explain(ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5)), model_config)
    estimates = {task.name: task.estimate for member in report.members for task in member.tasks}
    rows, hidden, rank, vocab, padded_vocab = 3, 8, 2, 32, 64
    assert estimates["predict"].matmul_flops_estimate == rows * 6 * hidden * rank
    assert estimates["predict"].projection_flops_estimate == rows * 2 * hidden * padded_vocab
    # Input hidden rows are constants: backward needs four GEMMs, forward needs three.
    assert estimates["loss"].matmul_flops_estimate == rows * 14 * hidden * rank
    assert estimates["loss"].optimizer_bytes_estimate > 0
    assert estimates["predict"].sampler_bytes_estimate == rows * vocab * 10
    dola = pllm.explain(DoLaWorkload(layers=(0, 2)), model_config)
    contrast = next(task for member in dola.members for task in member.tasks if task.name == "contrast")
    assert contrast.estimate.projection_flops_estimate == rows * 3 * 2 * hidden * padded_vocab
    assert contrast.estimate.sampler_bytes_estimate == rows * vocab * 12
    lora = pllm.explain(OutputLoRAWorkload(), model_config)
    assert any(
        task.estimate is not None and task.estimate.optimizer_bytes_estimate > 0
        for member in lora.members
        for task in member.tasks
    )
    assert "estimate / step" in str(report)
    assert all(field.dtype for state in report.states for field in state.fields)

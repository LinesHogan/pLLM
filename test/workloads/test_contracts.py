"""One CPU schedule contract table for the maintained model workloads."""

import json

import pytest
import torch
from safetensors.torch import save_file

import pllm
from pllm import testing
from pllm.workloads import (
    DecodeFactorsWorkload,
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
    PromptMixtureWorkload,
    RecirculationWorkload,
    ReFTConfig,
    ReFTWorkload,
    StepLoopWorkload,
    TLMConfig,
    TLMWorkload,
    TTRLWorkload,
)
from pllm.workloads.dola import DoLaWorkload
from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload
from pllm.workloads.state_producers import FactorAdapterWorkload
from pllm.workloads.video import VideoWorkload


@pytest.mark.parametrize(
    "workload",
    (
        DummyWorkload(),
        DoLaWorkload(layers=(0, 2, 4, 6, 8)),
        ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5)),
        FactorAdapterWorkload(),
        PromptFactorsWorkload(),
        PromptMixtureWorkload(),
        DecodeFactorsWorkload(),
        OutputLoRAWorkload(),
        OverRIDEWorkload(OverRIDEConfig(rank=2, num_iterations=3)),
        PrefillAdaptersWorkload(rank=2, chunk_size=4),
        GatedDeltaWorkload(GatedDeltaConfig((pllm.model.Projection("down_proj", 0),), head_width=4, chunk_size=4)),
        InPlaceTTTWorkload(InPlaceTTTConfig((pllm.model.Projection("down_proj", 0),), 4, 0.01), dtype=torch.bfloat16),
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
def test_builtin_schedule_contracts(workload, tmp_path):
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
    save_file(
        {
            "model.layers.0.mlp.ttt_conv.weight": torch.ones(8, 1, 5) / 5,
            "model.layers.0.mlp.ttt_proj.weight": torch.eye(8),
        },
        tmp_path / "model.safetensors",
    )
    result = testing.check(workload, pllm.ModelConfig(str(tmp_path)))
    assert result.passed or result.skipped

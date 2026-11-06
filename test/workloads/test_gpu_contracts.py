"""Public GPU contract checks over sampling, adapters, activation steering and exact training."""

from dataclasses import dataclass, replace
from functools import partial

import pytest
import torch

import pllm
from pllm.backends.vllm import runtime
from pllm.testing import check_on_gpu
from pllm.workloads import DoLaWorkload, ESampConfig, ESampWorkload, OutputLoRAWorkload, PromptFactorsWorkload
from pllm.workloads.state_producers import FactorAdapterWorkload
from test.gpu.vllm_adaptive_depth_smoke import Exit

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def fixed_kv_cache(monkeypatch):
    """Keep small contract models independent of free-memory profiling."""
    monkeypatch.setattr(runtime, "LLM", partial(runtime.LLM, kv_cache_memory_bytes=32 << 20))


@dataclass(frozen=True)
class Steering(pllm.Workload):
    """CAA's activation consumer with a zero direction as its neutral setting."""

    @property
    def direction(self) -> pllm.InstanceState:
        return pllm.InstanceState({"vector": lambda model: (model.hidden_size,)})

    @property
    def steer(self) -> pllm.model.ActivationPatch:
        return pllm.model.ActivationPatch("direction", (0,), operator="add", decode=True)


@pytest.mark.parametrize(
    "workload,neutral,capacity",
    [
        (ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5)), ESampWorkload(ESampConfig(rank=2)), None),
        (ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5)), ESampWorkload(ESampConfig(rank=2)), 4096),
        (OutputLoRAWorkload(), OutputLoRAWorkload(), None),
        (FactorAdapterWorkload(), FactorAdapterWorkload(), None),
        (Steering(), Steering(), None),
        (DoLaWorkload(layers=(0, 2)), None, None),
        (PromptFactorsWorkload(), None, None),
        # The pretrained LM head needs late-layer features. Midpoint exits produce near-tied junk tokens,
        # whose argmax changes with GEMM width after eviction even when recompute is correct.
        (Exit("mixed", exit_layer=22), Exit("all"), None),
    ],
    ids=("esamp", "esamp_bounded", "output_lora", "adapter", "caa", "dola", "prompt_factors", "adaptive"),
)
def test_gpu_contracts(workload, neutral, capacity):
    config = pllm.ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=64,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.25,
        execution=pllm.ExecutionOptions(capture_memory_bytes=256 << 20, guidance_capacity=capacity),
    )
    result = check_on_gpu(workload, config, neutral=neutral)
    assert "explain/probe" in result.passed
    print(type(workload).__name__, result)


def test_inplace_neutral_contract(tmp_path):
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from pllm.workloads import InPlaceTTTConfig, InPlaceTTTWorkload

    shape = Qwen3Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        vocab_size=151936,
        tie_word_embeddings=True,
    )
    torch.manual_seed(41)
    Qwen3ForCausalLM(shape).to(torch.bfloat16).save_pretrained(tmp_path)
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").save_pretrained(tmp_path)
    path = str(tmp_path / "model.safetensors")
    weights = load_file(path)
    weights.update(
        {
            f"model.layers.0.mlp.ttt_{name}.weight": torch.randn(size, dtype=torch.bfloat16) * 0.01
            for name, size in (("conv", (128, 1, 5)), ("proj", (128, 128)))
        }
    )
    save_file(weights, path)
    config = pllm.ModelConfig(
        str(tmp_path), max_model_len=64, max_num_seqs=2, max_num_batched_tokens=64, gpu_memory_utilization=0.25
    )
    options = InPlaceTTTConfig((pllm.model.Projection("down_proj", 0),), 4, 0.01)
    result = check_on_gpu(
        InPlaceTTTWorkload(options), config, neutral=InPlaceTTTWorkload(replace(options, learning_rate=0))
    )
    assert "exact decode: eager terminal request state on fixed per-token inputs" in result.passed
    assert "neutral: greedy and seeded sampling bitwise" in result.passed
    print(result)

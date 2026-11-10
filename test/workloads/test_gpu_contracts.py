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
        execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=256 << 20, guidance_capacity=capacity),
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
    assert "neutral: greedy and seeded sampling bitwise vs vanilla with the same kernels" in result.passed
    print(result)


@pytest.mark.parametrize("runner", ("mrv1", "mrv2"))
def test_distribution_contract(monkeypatch, runner):
    from pllm.testing import check_distribution
    from pllm.testing_distribution import SamplerObservation

    record = SamplerObservation.record

    def without_sync(self, sampled, logits):
        mode = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            return record(self, sampled, logits)
        finally:
            torch.cuda.set_sync_debug_mode(mode)

    monkeypatch.setattr(SamplerObservation, "record", without_sync)
    config = pllm.ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=64,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.25,
        runner=runner,
    )
    near = check_distribution(
        ESampWorkload(ESampConfig(rank=2, intervention_beta=1e-4)),
        config,
        temperature=0.8,
        max_mean_kl=0.01,
        max_p99_kl=0.01,
        max_mean_tv=0.01,
        max_p99_tv=0.05,
    )
    strong = check_distribution(DoLaWorkload(layers=(0, 2)), config, max_mean_kl=0.01, max_p99_kl=0.01)
    assert near.passed and not strong.passed
    assert near.token_count == strong.token_count == 32
    assert near.tv.mean < strong.tv.mean
    assert strong.kl_workload_to_plain.p99 > 0.01
    assert strong.tv.worst[0].value == strong.tv.max
    assert all(0 <= item.prompt_index < 2 and item.token_offset >= 1 for item in strong.tv.worst)
    print("near-neutral", near, near.run_seconds, near.elapsed_seconds)
    print("strong", strong, strong.run_seconds, strong.elapsed_seconds)

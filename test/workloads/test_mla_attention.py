"""Compressed attention numerics, DeepSeek model planning and packed CPU schedule contracts."""

import json
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

import pllm
from pllm import testing
from pllm.execution.attention import PagedKV, paged_latent_attention, selected_latent_attention
from pllm.execution.attention_methods import TraceRead
from test.workloads.attention_workloads import WindowAttention


def _batch(dtype):
    rng = torch.Generator().manual_seed(41)
    # Request one spans two attention tiles. Block zero is padding and contains NaNs.
    table = torch.randperm(36, generator=rng).reshape(2, 18) + 1
    cache = torch.randn(37, 8, 7, generator=rng).to(dtype)
    cache[0] = torch.nan
    uk = torch.randn(3, 5, 4, generator=rng).to(dtype)
    uv = torch.randn(3, 4, 6, generator=rng).to(dtype)
    q = torch.randn(5, 3, 8, generator=rng).to(dtype)
    kv = PagedKV.from_latent(
        cache[..., :4],
        cache[..., 4:],
        uk,
        uv,
        table,
        torch.tensor([9, 134]),
        torch.tensor([0, 2, 5]),
        len(q),
        TraceRead(134),
    )
    return q, kv


def _expanded_reference(q, kv, window, topk):
    output = q.new_zeros((*q.shape[:2], kv.v_head_dim))
    lse = q.new_full(q.shape[:2], -torch.inf, dtype=torch.float32)
    for row, (request, position) in enumerate(((0, 7), (0, 8), (1, 131), (1, 132), (1, 133))):
        pages = kv.block_table[request]
        latent = kv.latent_cache[pages].flatten(0, 1).double()
        rope = kv.rope_cache[pages].flatten(0, 1).double()
        keys = torch.cat(
            (torch.einsum("tl,hpl->htp", latent, kv.w_uk_t.double()), rope.expand(q.shape[1], -1, -1)), dim=-1
        )
        values = torch.einsum("tl,hlv->htv", latent, kv.w_uv.double())
        for head in range(q.shape[1]):
            tokens = torch.arange(position + 1)
            if topk is not None:
                chosen = topk[row] if topk.ndim == 2 else topk[row, head]
                tokens = tokens[torch.isin(tokens, chosen)]
            if kv.storage_window:
                tokens = tokens[(tokens > position - kv.storage_window) | (tokens < kv.prompt_lens[request])]
            if window:
                tokens = tokens[tokens > position - window]
            if len(tokens):
                query = q[row, head].double()
                key, value = keys[head, tokens], values[head, tokens]
                output[row, head] = F.scaled_dot_product_attention(query[None, None], key[None], value[None])[0, 0]
                lse[row, head] = (key @ query * q.shape[-1] ** -0.5).logsumexp(0)
    return output, lse


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("selection", ("full", "window", "shared", "per_head", "sink"))
def test_matrix_absorption(dtype, selection):
    q, kv = _batch(dtype)
    topk = None
    window = 4 if selection in ("window", "per_head") else 0
    if selection in ("shared", "per_head", "sink"):
        topk = torch.tensor([0, 1, 4, 7, 130, 132, -1, 999]).expand(len(q), -1).clone()
        topk[1] = -1
        if selection == "per_head":
            topk = topk[:, None].expand(-1, q.shape[1], -1).clone()
            topk[:, 0, :3] = -1
            topk[:, 2] = -1
        if selection == "sink":
            kv = replace(kv, storage_window=3, prompt_lens=torch.tensor([2, 1]))
    scale = q.shape[-1] ** -0.5
    actual = (
        paged_latent_attention(kv, q, scale, window)
        if topk is None
        else selected_latent_attention(kv, q, scale, window, topk)
    )
    torch.testing.assert_close(actual, _expanded_reference(q, kv, window, topk), atol=1e-4, rtol=1e-4)


@dataclass(frozen=True)
class _Monitor(pllm.Workload):
    @pllm.custom_attention(layers=(0,))
    def monitor(self, q, kv):
        out, lse = kv.attend(q)
        _, near = kv.attend(q * 0.5, window=3)
        return out, near - lse

    @pllm.export(max_rows=4, max_cols=4)
    def report(self, lse=pllm.attention(0, "lse"), aux=pllm.attention(0, "aux")):
        return torch.cat((lse, aux), dim=1)


@pytest.fixture
def mla_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                architectures=["DeepseekV2ForCausalLM"],
                model_type="deepseek_v2",
                hidden_size=64,
                num_hidden_layers=2,
                vocab_size=64,
                intermediate_size=64,
                num_attention_heads=2,
                kv_lora_rank=64,
                qk_nope_head_dim=32,
                qk_rope_head_dim=16,
                v_head_dim=32,
            )
        )
    )
    return pllm.ModelConfig(str(tmp_path), max_model_len=32, max_num_seqs=3, max_num_batched_tokens=16)


def test_mla_explain_and_check(mla_config):
    report = pllm.explain(_Monitor(), mla_config)
    shape = report.model
    assert (shape.head_dim, shape.v_head_dim, shape.kv_lora_rank) == (48, 32, 64)
    (cost,) = report.attention
    assert cost.kv_bytes_per_token == (64 + 16) * 2 * (32 + 3)
    assert cost.flops_per_token == 2 * 2 * (2 * 64 + 16) * (32 + 3)
    result = testing.check(_Monitor(), mla_config)
    assert "custom-attention method math / CPU paged KV schedules" in result.passed


def test_mla_rejects_window_storage(mla_config):
    with pytest.raises(ValueError, match="MLA @pllm.custom_attention supports keep='all'"):
        pllm.explain(WindowAttention(), mla_config)


def _mla_model(q, kv):
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV2MLAAttention

    # Build the native module path without initializing a model-parallel process group.
    model = DeepseekV2ForCausalLM.__new__(DeepseekV2ForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Module()
    layer = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([layer])
    layer.self_attn = DeepseekV2MLAAttention.__new__(DeepseekV2MLAAttention)
    torch.nn.Module.__init__(layer.self_attn)
    layer.self_attn.mla_attn = torch.nn.Module()
    mla = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(mla)
    layer.self_attn.mla_attn.mla_attn = mla
    mla.kv_cache_dtype, mla.kv_lora_rank, mla.num_heads, mla.v_head_dim = "auto", 4, 3, 6
    mla.scale, mla.impl = q.shape[-1] ** -0.5, SimpleNamespace(dcp_world_size=1)
    mla.W_UK_T, mla.W_UV = torch.nn.Parameter(kv.w_uk_t), torch.nn.Parameter(kv.w_uv)
    return model, mla


def _mixed_metadata(kv):
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonDecodeMetadata,
        MLACommonMetadata,
        MLACommonPrefillMetadata,
    )

    return MLACommonMetadata(
        num_reqs=2,
        max_query_len=3,
        max_seq_len=134,
        num_actual_tokens=5,
        query_start_loc=kv.query_start,
        slot_mapping=torch.arange(5),
        num_decodes=1,
        num_decode_tokens=2,
        num_prefills=1,
        decode=MLACommonDecodeMetadata(kv.block_table[:1], kv.seq_lens[:1], None),
        prefill=MLACommonPrefillMetadata(
            kv.block_table[1:],
            torch.tensor([0, 3]),
            3,
            chunked_context=SimpleNamespace(context_lens=torch.tensor([131])),
        ),
    )


@pytest.mark.parametrize("capture_lse", (False, True))
def test_mla_forward_hook(capture_lse):
    from pllm.backends.vllm.custom_attention import CustomAttention
    from pllm.execution.attention_methods import AttentionFact, AttentionMethod, AttentionProgram

    q, kv = _batch(torch.float32)
    model, mla = _mla_model(q, kv)
    cache = torch.cat((kv.latent_cache, kv.rope_cache), dim=-1)

    def native_read(query, key, rope, cache, metadata, *, output):
        output[: metadata.num_actual_tokens].fill_(17)
        return output

    native = mla.forward_impl = Mock(side_effect=native_read)
    program = AttentionProgram()
    program.add(AttentionMethod("monitor", (0,), _Monitor().monitor, ()))
    aux = program.facts[AttentionFact(0, "aux")] = torch.empty(5, 3)
    if capture_lse:
        program.facts[AttentionFact(0, "lse")] = torch.empty(5, 3)
    sites = SimpleNamespace(store_rows=torch.Tensor.copy_)
    hooks = CustomAttention(SimpleNamespace(model=model), program, sites)
    output = q.new_full((7, 18), -2)
    try:
        assert mla.forward_impl(q, q, q, cache, None, output).count_nonzero() == 0
        mla.forward_impl(q, q, q, cache, _mixed_metadata(kv), output)
        reference, lse = paged_latent_attention(kv, q, mla.scale)
        near = paged_latent_attention(kv, q * 0.5, mla.scale, 3)[1]
        torch.testing.assert_close(aux, near - lse)
        if capture_lse:
            torch.testing.assert_close(program.facts[AttentionFact(0, "lse")], lse)
            torch.testing.assert_close(output[:5], reference.flatten(1))
            native.assert_not_called()
        else:
            native.assert_called_once()
            assert (output[:5] == 17).all()
        assert output[5:].count_nonzero() == 0
    finally:
        hooks.close()
    assert mla.forward_impl is native


@dataclass(frozen=True)
class _StatelessLseLogits(pllm.Workload):
    @pllm.custom_attention(layers=(0,))
    def attend(self, q, kv):
        return kv.attend(q)[0]

    @pllm.add_logits
    def bias(self, lse=pllm.attention(0, "lse")):
        return lse.sum(dim=-1, keepdim=True).expand(-1, 64) * 0.0


@dataclass(frozen=True)
class _StatelessSampleBias(pllm.Workload):
    @pllm.add_logits
    def bias(self, index=pllm.sample_index()):
        return index[:, None].float().expand(-1, 64) * 0.0


def test_stateless_attention_fact_and_sample_index_row_sizing(mla_config):
    from pllm.config import ExecutionOptions
    from pllm.contracts.semantic_graph import TaskSymbol
    from pllm.testing_contracts import contract_facts

    _, facts = contract_facts(_Monitor(), mla_config)
    facts, _ = _Monitor().specialize(facts, ExecutionOptions())
    resources = {item.task: item for item in facts.timeline.task_resources}
    assert resources[TaskSymbol("__attention_0_lse")].input_shapes == ((mla_config.max_num_seqs, 64),)
    assert resources[TaskSymbol("__attention_0_aux")].input_shapes == ((mla_config.max_num_seqs, 64),)
    assert pllm.explain(_StatelessLseLogits(), mla_config).members
    assert pllm.explain(_StatelessSampleBias(), mla_config).members

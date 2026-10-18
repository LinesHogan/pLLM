"""Attention resource binding needs vLLM endpoints, without a model wrapper."""

from dataclasses import replace

import pytest
import torch
from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import ForwardContext
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionMetadata

from pllm.backends.vllm.region_attention import (
    attention_paged_access,
    select_attention_metadata,
    validate_attention_resources,
)
from test.boundary.test_paged_access import rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM FlashAttention backend requires CUDA")
def test_original_attention_resource_binds_without_a_model():
    config = VllmConfig(attention_config=AttentionConfig(flash_attn_version=2))
    with set_current_vllm_config(config):
        attention = Attention(
            2, 64, 0.125, num_kv_heads=1, prefix="independent.attention", attn_backend=FlashAttentionBackend
        )
    # Resource-only check: real backend object, CPU storage, no attention forward.
    attention.kv_cache = torch.zeros(4, 1, 16, 128)
    validate_attention_resources((attention,), hidden_size=128)
    slots = torch.tensor([16, 17, 32, 33])
    metadata = FlashAttentionMetadata(
        4,
        2,
        torch.tensor([0, 2, 4], dtype=torch.int32),
        18,
        torch.tensor([18, 18], dtype=torch.int32),
        torch.tensor([[0, 1], [0, 2]]),
        slots,
        False,
        0,
        None,
        None,
        None,
    )
    name = attention.layer_name
    context = ForwardContext({name: attention}, {name: metadata}, {name: slots})
    batch = rows((16, 17, 16, 17))
    access = attention_paged_access(context, (attention,), batch)
    selected, cache, block, offset = select_attention_metadata(
        context, attention, batch[2], 17, torch.tensor([0, 1, 17], dtype=torch.int32), access
    )
    assert cache is attention.kv_cache and (block, offset) == (2, 0)
    assert selected.slot_mapping.data_ptr() == slots[2:].data_ptr()
    assert selected.block_table.data_ptr() == metadata.block_table[1:].data_ptr()
    assert selected.seq_lens.tolist() == [17] and selected.num_actual_tokens == 1
    assert context.attn_metadata[name] is metadata and metadata.num_actual_tokens == 4
    assert torch.count_nonzero(cache) == 0
    with pytest.raises(ValueError, match="different slot mappings"):
        attention_paged_access(replace(context, slot_mapping={name: slots + 1}), (attention,), batch)
    with pytest.raises(ValueError, match="ordinary causal"):
        attention_paged_access(
            replace(context, attn_metadata={name: replace(metadata, causal=False)}), (attention,), batch
        )
    attention.kv_cache_dtype = "fp8"
    with pytest.raises(ValueError, match="unquantized"):
        validate_attention_resources((attention,), hidden_size=128)

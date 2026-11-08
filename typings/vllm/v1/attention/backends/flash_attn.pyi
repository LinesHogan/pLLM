"""The FlashAttention metadata fields and impl attributes shared by vLLM 0.27.1 and 0.32.0."""

from dataclasses import dataclass

import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionMetadataBuilder, CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

@dataclass
class FlashAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_dcp_context_kv_len: int | None = ...
    dcp_context_kv_lens: torch.Tensor | None = ...
    num_decode_reqs: int = ...
    num_prefill_reqs: int = ...
    num_decode_tokens: int = ...
    num_prefill_tokens: int = ...
    scheduler_metadata: torch.Tensor | None = ...
    max_num_splits: int = ...
    causal: bool | torch.Tensor = ...
    use_cascade: bool = ...
    rswa_prefix_lens: torch.Tensor | None = ...
    rswa_window: int | None = ...
    rswa_window_tensor: torch.Tensor | None = ...

class FlashAttentionImpl:
    alibi_slopes: torch.Tensor | None
    logits_soft_cap: float
    sinks: torch.Tensor | None
    scale: float
    sliding_window: tuple[int, int]
    vllm_flash_attn_version: int | None

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None: ...
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    def __init__(
        self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device: torch.device
    ) -> None: ...
    def build(
        self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False
    ) -> FlashAttentionMetadata: ...

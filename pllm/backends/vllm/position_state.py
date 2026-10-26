"""Register block-sized positional caches and stage their own group mappings for routes."""

import torch
from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext
from vllm.model_executor.models.extract_hidden_states import (
    CacheOnlyAttentionBackend,
    CacheOnlyAttentionLayer,
    CacheOnlyAttentionMetadata,
    CacheOnlyAttentionMetadataBuilder,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.execution.position_state import PositionBuffers

POSITION_ENV = "PLLM_POSITION_STATES"


class PositionMetadata(CacheOnlyAttentionMetadata):
    """Keep the cache-only group's own predecessor table in the forward context."""

    def __init__(self, common: CommonAttentionMetadata) -> None:
        super().__init__(common.slot_mapping)
        self.block_table = common.block_table_tensor


class PositionMetadataBuilder(CacheOnlyAttentionMetadataBuilder):
    """Expose device mappings without introducing an attention computation."""

    def build(
        self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False
    ) -> PositionMetadata:
        return PositionMetadata(common_attn_metadata)


class PositionBackend(CacheOnlyAttentionBackend):
    """Use vLLM's cache-only allocation shape with positional metadata."""

    @staticmethod
    def get_builder_cls() -> type[PositionMetadataBuilder]:
        return PositionMetadataBuilder


class PositionCacheLayer(CacheOnlyAttentionLayer):
    """One H-vector per allocation block, budgeted and transferred as a KV layer."""

    def get_attn_backend(self) -> type[PositionBackend]:
        return PositionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> HiddenStateCacheSpec:
        allocation = vllm_config.cache_config.block_size
        if self.head_size % allocation:
            raise ValueError("PositionState needs hidden_size divisible by the KV allocation block size")
        return HiddenStateCacheSpec(
            block_size=allocation,
            num_kv_heads=1,
            head_size=self.head_size // allocation,
            dtype=self.kv_cache_torch_dtype,
        )


class PositionCacheProvider(PositionBuffers):
    """Stage allocation-page predecessor indices, never reading page-table values on host."""

    def __init__(
        self, layers: tuple[PositionCacheLayer, ...], initial: tuple[float, ...], allocation: int, tensor: torch.Tensor
    ) -> None:
        self.layers, self.allocation = layers, allocation
        if any(
            not layer.kv_cache.is_contiguous()
            or allocation % layer.kv_cache.shape[1]
            or layer.kv_cache.shape[0] % (allocation // layer.kv_cache.shape[1])
            or layer.kv_cache.untyped_storage().nbytes() != layer.kv_cache.numel() * layer.kv_cache.element_size()
            or layer.kv_cache.dtype != tensor.dtype
            or layer.kv_cache.shape[1:] != (layer.kv_cache.shape[1], 1, tensor.shape[1] // allocation)
            for layer in layers
        ):
            raise ValueError(
                "PositionState requires contiguous unpadded model-dtype cache pages and integral allocation/kernel blocks"
            )
        super().__init__(tuple(layer.kv_cache.view(-1, tensor.shape[1]) for layer in layers), initial, tensor)

    def stage(self, context: ForwardContext, row: PackedPhaseRow) -> None:
        """Use each registered layer's mapping, including device-side page permutations."""
        assert isinstance(context.attn_metadata, dict) and isinstance(context.slot_mapping, dict)
        self.flags[0].fill_(row.token_offset == 0)
        self.flags[1].fill_(row.phase is HiddenPhase.DECODE)
        for layer, indices in zip(self.layers, self.indices, strict=True):
            metadata = context.attn_metadata[layer.layer_name]
            assert isinstance(metadata, PositionMetadata)
            slot = context.slot_mapping[layer.layer_name][row.packed_row_index : row.packed_row_index + 1]
            current = slot // self.allocation
            kernel = layer.kv_cache.shape[1]
            previous = (
                current
                if row.token_offset % self.allocation
                else metadata.block_table[row.request_ordinal, max(0, row.token_offset - 1) // kernel].reshape(1)
                // (self.allocation // kernel)
            )
            indices[0].copy_(current)
            indices[1].copy_(previous)

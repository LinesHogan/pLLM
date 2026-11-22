"""Use vLLM's FlashInfer plans for stored windows in custom attention.

FA2 disables its packed GQA decode for windows. On Ampere/Ada FlashInfer keeps head grouping,
while vLLM owns page-index staging, split planning and CUDA-graph buffers.
The FA2 metadata also serves authored reads with a different window, heads or head stride.
Sessions load this module after worker startup: importing FlashInfer can initialize CUDA,
so its release-specific metadata adapters also stay here.
"""

import math
from dataclasses import dataclass, replace

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from vllm import envs
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.flashinfer import (
    FIDecode,
    FIPrefill,
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadata,
    FlashInferMetadataBuilder as VllmFlashInferMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from pllm.backends.vllm.compat import VLLM_027, VLLM_028_030, VLLM_MINOR

if VLLM_027:
    from vllm.v1.attention.backends.utils import get_kv_cache_layout

    def flashinfer_head_first(metadata: FlashInferMetadata) -> bool:
        """Match 0.27's FlashInfer wrappers to their process-wide cache layout."""
        return get_kv_cache_layout() == "HND"

    class FlashInferMetadataBuilder(VllmFlashInferMetadataBuilder):
        """Give 0.27's planner the exact host lengths that >=0.31 already uses without a CUDA copy."""

        def build(
            self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False
        ) -> FlashInferMetadata:
            # MRv2's host bound is exact outside speculative decoding, which pLLM does not enable.
            if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
                common_attn_metadata = replace(
                    common_attn_metadata, _seq_lens_cpu=common_attn_metadata.seq_lens_cpu_upper_bound
                )
            return super().build(common_prefix_len, common_attn_metadata, fast_build)

    @dataclass
    class WindowMetadata(FlashInferMetadata):
        """Native FlashInfer execution plus FA2 metadata for authored alternative reads."""

        flash: FlashAttentionMetadata

elif VLLM_028_030:
    if VLLM_MINOR < (0, 29):
        from vllm.v1.attention.backends.utils import get_kv_cache_layout

        def _builder_head_first(builder: VllmFlashInferMetadataBuilder) -> bool:
            return get_kv_cache_layout() == "HND"
    else:
        from vllm.v1.attention.backends.utils import get_flashinfer_layout_string

        def _builder_head_first(builder: VllmFlashInferMetadataBuilder) -> bool:
            return get_flashinfer_layout_string(builder.kv_cache_layout) == "HND"

    class FlashInferMetadataBuilder(VllmFlashInferMetadataBuilder):
        """Give 0.28..0.30's planner the exact host lengths without a CUDA copy."""

        def build(
            self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False
        ) -> FlashInferMetadata:
            if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
                common_attn_metadata = replace(
                    common_attn_metadata, _seq_lens_cpu=common_attn_metadata.seq_lens_cpu_upper_bound
                )
            return super().build(common_prefix_len, common_attn_metadata, fast_build)

    @dataclass
    class WindowMetadata(FlashInferMetadata):
        """Native FlashInfer execution plus FA2 metadata for authored alternative reads."""

        flash: FlashAttentionMetadata
        head_first: bool = False

    def flashinfer_head_first(metadata: WindowMetadata) -> bool:
        """Match 0.28..0.30 FlashInfer wrappers to the resolved cache layout."""
        return metadata.head_first

else:
    from vllm.v1.attention.backends.utils import get_flashinfer_layout_string

    def flashinfer_head_first(metadata: FlashInferMetadata) -> bool:
        """Match >=0.31's FlashInfer wrappers to the layout carried in their metadata."""
        return get_flashinfer_layout_string(metadata.kv_cache_layout) == "HND"

    FlashInferMetadataBuilder = VllmFlashInferMetadataBuilder

    @dataclass
    class WindowMetadata(FlashInferMetadata):
        """Native FlashInfer execution plus FA2 metadata for authored alternative reads."""

        flash: FlashAttentionMetadata


class WindowMetadataBuilder(FlashInferMetadataBuilder):
    """Let both upstream builders consume the same native page mappings."""

    def __init__(
        self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device: torch.device
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        assert vllm_config.speculative_config is None
        self.flash = FlashAttentionMetadataBuilder(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False
    ) -> WindowMetadata:
        infer = super().build(common_prefix_len, common_attn_metadata, fast_build)
        flash = self.flash.build(common_prefix_len, common_attn_metadata, fast_build)
        if VLLM_028_030:
            return WindowMetadata(**vars(infer), flash=flash, head_first=_builder_head_first(self))
        return WindowMetadata(**vars(infer), flash=flash)


class WindowBackend(FlashInferBackend):
    """Keep upstream FlashInfer storage and execution, with metadata for authored reads."""

    @staticmethod
    def get_builder_cls() -> type[WindowMetadataBuilder]:
        return WindowMetadataBuilder


def window_attention(
    metadata: WindowMetadata, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, output: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read PagedKV's NHD caches with vLLM's planned wrappers and return natural-log LSE."""
    if flashinfer_head_first(metadata):
        keys, values = keys.transpose(1, 2), values.transpose(1, 2)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    count = metadata.num_decode_tokens
    if metadata.decode is not None:
        assert isinstance(metadata.decode, FIDecode)
        metadata.decode.wrapper.run(  # pyright: ignore[reportUnknownMemberType]
            q[:count], (keys, values), out=output[:count], lse=lse[:count], return_lse=True
        )
    if metadata.prefill is not None:
        assert isinstance(metadata.prefill, FIPrefill)
        assert isinstance(metadata.prefill.wrapper, BatchPrefillWithPagedKVCacheWrapper)
        metadata.prefill.wrapper.run(  # pyright: ignore[reportUnknownMemberType]
            q[count:], (keys, values), out=output[count:], lse=lse[count:], return_lse=True
        )
    # FlashInfer reports base-2 LSE; authored reads expose natural logarithms.
    return output, lse.mul_(math.log(2))


def install_window(attention: Attention, window: int) -> None:
    """Select native FlashInfer windows on Ampere/Ada; keep FA2 on other architectures."""
    attention.sliding_window = window
    capability = current_platform.get_device_capability()
    if (
        capability is None
        or capability.major != 8
        or envs.VLLM_BATCH_INVARIANT
        or not WindowBackend.supports_head_size(attention.head_size)
    ):
        return
    impl = attention_impl(attention)
    assert isinstance(impl, FlashAttentionImpl)
    attention.attn_backend = WindowBackend
    attention.impl = FlashInferImpl(
        attention.num_heads,
        attention.head_size,
        impl.scale,
        attention.num_kv_heads,
        None,
        window,
        attention.kv_cache_dtype,
        impl.logits_soft_cap,
    )


def attention_impl(attention: Attention) -> FlashAttentionImpl | FlashInferImpl:
    """Admit ordinary FA2 layers and the stored-window backend installed here."""
    impl = attention.impl
    if isinstance(impl, FlashInferImpl) and attention.attn_backend is WindowBackend:
        pass
    elif not isinstance(impl, FlashAttentionImpl) or impl.vllm_flash_attn_version != 2:
        raise ValueError(
            "@pllm.custom_attention needs vLLM's FlashAttention 2 backend; set VLLM_ATTENTION_BACKEND=FLASH_ATTN and VLLM_FLASH_ATTN_VERSION=2"
        )
    if impl.alibi_slopes is not None or impl.logits_soft_cap != 0 or impl.sinks is not None:
        raise ValueError("@pllm.custom_attention needs ordinary RoPE attention without ALiBi, softcap or sinks")
    return impl

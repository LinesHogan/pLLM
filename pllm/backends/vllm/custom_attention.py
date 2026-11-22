"""Install authored dense and MLA reads after vLLM writes the current tokens to its paged cache.

The upstream op already supplies graph ordering and capture. Full reads use FA2; stored windows use FlashInfer; prompt-sink reads use vLLM's Triton path.
Selected reads gather logical positions without changing block tables.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import torch
from torch.utils._pytree import tree_map_only
from vllm.distributed.parallel_state import get_dcp_group
from vllm.model_executor.layers.attention.mla_attention import MLACommonDecodeMetadata, MLACommonMetadata
from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata
from vllm.vllm_flash_attn import flash_attn_varlen_func

from pllm.backends.vllm.architecture import Decoder
from pllm.backends.vllm.compat import VLLM_027, VLLM_028_030, VLLM_031
from pllm.backends.vllm.model_sites import ModelSites
from pllm.backends.vllm.rswa_attention import rswa_attention
from pllm.contracts.kv_layout import storage_layout
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.execution.attention import LatentAttention, PagedKV, selected_attention
from pllm.execution.attention_methods import AttentionFact, AttentionProgram
from pllm.execution.cuda_stream import upload
from pllm.execution.model_sites import DepthInput, StateInput

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionImpl
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl, FlashInferMetadata

    from pllm.backends.vllm.flashinfer_attention import WindowMetadata


@dataclass
class FlashRead:
    """One invocation's native metadata and first read's LSE, exposed as the layer's fact."""

    scale: float
    metadata: FlashAttentionMetadata
    output: torch.Tensor
    infer: WindowMetadata | None = None
    kv: PagedKV | None = None
    lse: torch.Tensor | None = None

    def __call__(self, q: torch.Tensor, window: int, topk: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.kv is not None
        output = (
            self.output
            if self.lse is None and q.shape == self.output.shape and q.dtype == self.output.dtype
            else torch.empty_like(q, memory_format=torch.contiguous_format)
        )
        if (
            self.kv.prompt_lens is not None
            and topk is None
            and (window > self.kv.storage_window or q.dtype != self.kv.key_cache.dtype)
        ):
            # Native R-SWA does not express a wider-window intersection or mixed query/cache precision.
            topk = torch.arange(self.metadata.max_seq_len, device=q.device)[None, :].expand(len(q), -1)
        if topk is not None:
            result = selected_attention(self.kv, q, self.scale, window, topk)
        elif self.kv.prompt_lens is not None and not window and q.dtype == self.kv.key_cache.dtype:
            result = rswa_attention(
                self.kv, q, self.scale, output, self.metadata.max_query_len, self.metadata.max_seq_len
            )
        elif (
            self.infer is not None
            and window == self.kv.storage_window
            and q.shape[1] == self.output.shape[1]
            and q.dtype == self.kv.key_cache.dtype
            and q.stride(-1) == 1
        ):
            from pllm.backends.vllm.flashinfer_attention import window_attention

            result = window_attention(self.infer, q, self.kv.key_cache, self.kv.value_cache, output)
        else:
            out, lse = flash_attn_varlen_func(
                q=q,
                out=output,
                k=self.kv.key_cache,
                v=self.kv.value_cache,
                cu_seqlens_q=self.metadata.query_start_loc,
                max_seqlen_q=self.metadata.max_query_len,
                seqused_k=self.metadata.seq_lens,
                max_seqlen_k=self.metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=None,
                window_size=[window - 1, 0] if window else [-1, -1],
                block_table=self.metadata.block_table,
                softcap=0.0,
                scheduler_metadata=self.metadata.scheduler_metadata,
                fa_version=2,
                num_splits=self.metadata.max_num_splits,
                return_softmax_lse=True,
            )
            result = out, lse.T
        if self.lse is None:
            self.lse = result[1]
        return result


class _MLALayer(Protocol):
    """The MLA forward boundary with its decode metadata type bound."""

    forward_impl: Callable[..., torch.Tensor]
    kv_lora_rank: int
    num_heads: int
    v_head_dim: int
    scale: float

    @property
    def impl(self) -> AttentionImpl[MLACommonMetadata[MLACommonDecodeMetadata]]: ...

    def get_parameter(self, target: str) -> torch.Tensor: ...


class CustomAttention:
    """A session's attention hooks; restore native implementations when the session closes."""

    def __init__(self, decoder: Decoder, program: AttentionProgram, sites: ModelSites) -> None:
        self.program, self.sites = program, sites
        self.installed: list[tuple[FlashAttentionImpl | FlashInferImpl, Callable[..., torch.Tensor]]] = []
        self.mla_hooks: list[MLAAttentionHook] = []
        try:
            if isinstance(decoder.model, DeepseekV2ForCausalLM):
                from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention

                self.program.validate_mla_layouts()
                for layer in sorted(program.methods):
                    attn = decoder.model.model.layers[layer].self_attn
                    if not isinstance(attn, DeepseekV2MLAAttention):
                        self._install_dense(decoder)
                        return
                    attention = attn.mla_attn.mla_attn
                    if attention.kv_cache_dtype != "auto":
                        raise ValueError("@pllm.custom_attention needs unquantized KV; set kv_cache_dtype=auto")
                    self.mla_hooks.append(MLAAttentionHook.install(self, layer, attention))
            else:
                self._install_dense(decoder)
        except Exception:
            self.close()
            raise

    def _install_dense(self, decoder: Decoder) -> None:
        from pllm.backends.vllm.flashinfer_attention import attention_impl

        program = self.program
        model = decoder.kv_writable("@pllm.custom_attention")
        for layer in sorted(program.methods):
            attention = model.model.layers[layer].self_attn.attn
            impl = attention_impl(attention)
            layout = storage_layout(program.layouts, layer)
            sharing = any(item.layer == layer and item.keep == "none" for item in program.layouts)
            window = layout.window if layout.keep == "window" and not layout.sink and not sharing else None
            if attention.sliding_window != window or (attention.kv_sharing_target_layer_name is not None) != sharing:
                raise ValueError("custom_attention needs full attention or a layout declared with keep/window/source")
            if attention.kv_cache_dtype != "auto":
                raise ValueError("@pllm.custom_attention needs unquantized KV; set kv_cache_dtype=auto")
            self.installed.append((impl, impl.forward))
            impl.forward = AttentionHook(self, impl.scale, layer).forward

    def apply(self, layer: int, q: torch.Tensor, kv: PagedKV, read: FlashRead | MLARead, output: torch.Tensor) -> None:
        """Run the method and publish requested facts over its packed rows."""
        self.sites.count = len(q)
        program, sites = self.program, self.sites
        method = program.methods[layer]
        kwargs = {
            name: source.value(sites, layer)
            if isinstance(source, (StateInput, DepthInput))
            else sites.current_rows(program.facts[source])
            for name, source in method.inputs
        }
        result = method.function(q, kv, **kwargs)
        out, aux = result if isinstance(result, tuple) else (result, None)
        output.copy_(out)
        for source, tensor in ((AttentionFact(layer, "lse"), read.lse), (AttentionFact(layer, "aux"), aux)):
            if source in program.facts:
                assert tensor is not None
                sites.store_rows(program.facts[source], tensor)

    def prepare(self, rows: PackedPhaseRowBatch, sampler: PackedSamplerRowBatch) -> None:
        """Stage sampled-to-packed indices before model and fact-snapshot tasks run."""
        if self.program.prompt_lens.numel():
            self.program.prompt_lens[: len(rows.request_indices)].copy_(
                upload(
                    rows.prompt_lengths[rows.request_indices],
                    self.program.prompt_lens.device,
                )
            )
        if self.program.sample_rows.numel():
            mapping = sampler.packed_indices
            self.program.sample_rows[: len(mapping)].copy_(upload(mapping, self.program.sample_rows.device))

    def close(self) -> None:
        """Restore the upstream reads before destroying the session scratch."""
        for impl, forward in reversed(self.installed):
            impl.forward = forward
        self.installed.clear()
        for hook in self.mla_hooks:
            hook.mla.forward_impl = hook.native
        self.mla_hooks.clear()


@dataclass
class AttentionHook:
    """One layer's read at the upstream attention boundary; native cache writes precede it."""

    owner: CustomAttention
    scale: float
    layer: int

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata | FlashInferMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output_scale is None and output_block_scale is None
        if attn_metadata is None:
            return output.zero_()
        from pllm.backends.vllm.flashinfer_attention import WindowMetadata

        infer = attn_metadata if isinstance(attn_metadata, WindowMetadata) else None
        metadata = infer.flash if infer is not None else attn_metadata
        assert isinstance(metadata, FlashAttentionMetadata)
        if (
            (VLLM_027 or VLLM_028_030 or VLLM_031)
            and metadata.use_cascade
            or metadata.causal is not True
            or metadata.max_dcp_context_kv_len not in (None, 0)
        ):
            raise ValueError(
                "@pllm.custom_attention needs ordinary causal FlashAttention metadata; disable cascade, DCP and prefix caching"
            )
        count = metadata.num_actual_tokens
        q = query[:count]
        keys, values = kv_cache.transpose(1, 2).chunk(2, dim=-1)
        read = FlashRead(self.scale, metadata, output[:count], infer)
        layout = storage_layout(self.owner.program.layouts, self.layer)
        kv = PagedKV(
            keys,
            values,
            metadata.block_table,
            metadata.seq_lens,
            metadata.query_start_loc,
            count,
            read,
            layout.window,
            self.owner.program.prompt_lens[: len(metadata.seq_lens)] if layout.sink else None,
        )
        read.kv = kv
        self.owner.apply(self.layer, q, kv, read, output[:count])
        return output


def _mla_pages(metadata: MLACommonMetadata[MLACommonDecodeMetadata]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode requests precede prefills. Prefill offsets exclude the cached prefix."""
    starts = metadata.query_start_loc[: metadata.num_reqs + 1]
    decode, prefill = metadata.decode, metadata.prefill
    if prefill is None:
        assert decode is not None
        return decode.block_table, decode.seq_lens, starts
    table = prefill.block_table[: metadata.num_prefills]
    lengths = prefill.query_start_loc[: metadata.num_prefills + 1].diff()
    if prefill.chunked_context is not None:
        lengths = lengths + prefill.chunked_context.context_lens
    if decode is None:
        return table, lengths, starts
    return torch.cat((decode.block_table, table)), torch.cat((decode.seq_lens, lengths)), starts


_R = TypeVar("_R")


class _NativeLSE(torch.Tensor):
    """vLLM returns output alone. Materialize the normalizer if workload math reads it."""

    compute: Callable[[], torch.Tensor]
    value: torch.Tensor | None

    @classmethod
    def deferred(cls, q: torch.Tensor, compute: Callable[[], torch.Tensor]) -> _NativeLSE:
        result = torch.empty(q.shape[:2], device="meta", dtype=torch.float32).as_subclass(cls)
        result.compute = compute
        result.value = None
        return result

    def materialize(self) -> torch.Tensor:
        if self.value is None:
            self.value = self.compute()
        return self.value

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., _R],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> _R:
        values, keywords = tree_map_only(_NativeLSE, _NativeLSE.materialize, (args, kwargs or {}))
        return func(*values, **keywords)


@dataclass
class MLARead:
    """Delegate unchanged queries to vLLM and expose compressed reads over the same pages."""

    query: torch.Tensor
    native: Callable[..., torch.Tensor]
    output: torch.Tensor
    need_lse: bool
    scale: float
    max_seq_len: int
    kv: PagedKV | None = None
    lse: torch.Tensor | None = None

    def normalizer(self) -> torch.Tensor:
        """Compute the LSE omitted by the native output-only kernel."""
        assert self.kv is not None
        return LatentAttention(self.kv, self.scale, self.max_seq_len)(self.query, 0, None)[1]

    def __call__(self, q: torch.Tensor, window: int, topk: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.kv is not None
        if q is self.query and not window and topk is None and (not self.need_lse or self.lse is not None):
            buffer = self.output if self.lse is None else torch.empty_like(self.output)
            out = self.native(output=buffer)[: len(q)].view(len(q), q.shape[1], self.kv.v_head_dim)
            lse = _NativeLSE.deferred(q, self.normalizer)
        else:
            out, lse = LatentAttention(self.kv, self.scale, self.max_seq_len)(q, window, topk)
        if self.lse is None:
            self.lse = lse
        return out, lse


@dataclass
class MLAAttentionHook:
    """Replace the already-bound forward_impl used by both MLA call paths."""

    owner: CustomAttention
    layer: int
    mla: _MLALayer
    native: Callable[..., torch.Tensor]

    @classmethod
    def install(cls, owner: CustomAttention, layer: int, mla: _MLALayer) -> MLAAttentionHook:
        """Save the bound method and install the post-write read for this layer."""
        hook = cls(owner, layer, mla, mla.forward_impl)
        mla.forward_impl = hook.forward_impl
        return hook

    def forward_impl(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata[MLACommonDecodeMetadata] | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        quant_group_size: int | None = None,
        quant_scale_ue8m0: bool | None = None,
        quant_col_major: bool | None = None,
        quant_tma_aligned: bool | None = None,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.zero_()
        if output_scale is not None or output_block_scale is not None or output.dtype != q.dtype:
            raise ValueError("@pllm.custom_attention needs model-dtype MLA output; disable fused output quantization")
        # MLACommon builders never select cascade. vLLM 0.27 resolves the DCP group on its first read.
        dcp_size = self.mla.impl.dcp_world_size
        if dcp_size == -1:
            dcp_size = get_dcp_group().world_size
        if (
            attn_metadata.causal is not True
            or dcp_size > 1
            or q_dcp_replicated is not None
            or (attn_metadata.decode is not None and attn_metadata.decode.dcp_tot_seq_lens is not None)
        ):
            raise ValueError("@pllm.custom_attention needs ordinary causal MLA metadata; disable cascade and DCP")
        count, rank = attn_metadata.num_actual_tokens, self.mla.kv_lora_rank
        query = q[:count]
        read = MLARead(
            query,
            partial(self.native, q, k_c_normed, k_pe, kv_cache, attn_metadata),
            output,
            AttentionFact(self.layer, "lse") in self.owner.program.facts,
            self.mla.scale,
            attn_metadata.max_seq_len,
        )
        kv = PagedKV.from_latent(
            kv_cache[..., :rank],
            kv_cache[..., rank:],
            self.mla.get_parameter("W_UK_T"),
            self.mla.get_parameter("W_UV"),
            *_mla_pages(attn_metadata),
            count,
            read,
        )
        read.kv = kv
        try:
            self.owner.apply(
                self.layer, query, kv, read, output[:count].view(count, self.mla.num_heads, self.mla.v_head_dim)
            )
        finally:
            read.kv = None
            read.lse = None
        return output

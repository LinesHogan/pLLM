"""Install authored dense reads inside vLLM's opaque attention op, after its native KV write.

The upstream op already supplies graph ordering and capture. Full reads use FA2; local and prompt-sink reads use vLLM's native Triton kernel.
Selected reads gather logical positions without changing block tables.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata
from vllm.vllm_flash_attn import flash_attn_varlen_func

from pllm.backends.vllm.architecture import Decoder
from pllm.backends.vllm.compat import VLLM_027
from pllm.backends.vllm.model_sites import ModelSites
from pllm.backends.vllm.window_attention import window_attention
from pllm.contracts.kv_layout import storage_layout
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.execution.attention import PagedKV, selected_attention
from pllm.execution.attention_methods import AttentionFact, AttentionProgram
from pllm.execution.cuda_stream import upload
from pllm.execution.model_sites import DepthInput, StateInput


@dataclass
class FlashRead:
    """One invocation's FA2 metadata and first read's LSE, exposed as the layer's fact."""

    impl: FlashAttentionImpl
    metadata: FlashAttentionMetadata
    output: torch.Tensor
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
            result = selected_attention(self.kv, q, self.impl.scale, window, topk)
        elif (window or self.kv.prompt_lens is not None) and q.dtype == self.kv.key_cache.dtype:
            result = window_attention(self.kv, q, self.impl.scale, window, output)
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
                softmax_scale=self.impl.scale,
                causal=True,
                alibi_slopes=self.impl.alibi_slopes,
                window_size=[window - 1, 0] if window else [-1, -1],
                block_table=self.metadata.block_table,
                softcap=self.impl.logits_soft_cap,
                scheduler_metadata=self.metadata.scheduler_metadata,
                fa_version=2,
                num_splits=self.metadata.max_num_splits,
                return_softmax_lse=True,
            )
            result = out, lse.T
        if self.lse is None:
            self.lse = result[1]
        return result


class CustomAttention:
    """A session's attention hooks; restore native implementations when the session closes."""

    def __init__(self, decoder: Decoder, program: AttentionProgram, sites: ModelSites) -> None:
        self.program, self.sites = program, sites
        self.installed: list[tuple[FlashAttentionImpl, Callable[..., torch.Tensor]]] = []
        model = decoder.kv_writable("@pllm.custom_attention")
        for layer in sorted(program.methods):
            attention = model.model.layers[layer].self_attn.attn
            impl = attention.impl
            if not isinstance(impl, FlashAttentionImpl) or impl.vllm_flash_attn_version != 2:
                raise ValueError(
                    "@pllm.custom_attention needs vLLM's FlashAttention 2 backend; set VLLM_ATTENTION_BACKEND=FLASH_ATTN and VLLM_FLASH_ATTN_VERSION=2"
                )
            layout = storage_layout(program.layouts, layer)
            sharing = any(item.layer == layer and item.keep == "none" for item in program.layouts)
            window = layout.window if layout.keep == "window" and not layout.sink and not sharing else None
            if attention.sliding_window != window or (attention.kv_sharing_target_layer_name is not None) != sharing:
                raise ValueError("custom_attention needs full attention or a layout declared with keep/window/source")
            if attention.kv_cache_dtype != "auto":
                raise ValueError("@pllm.custom_attention needs unquantized KV; set kv_cache_dtype=auto")
            if impl.alibi_slopes is not None or impl.logits_soft_cap != 0 or impl.sinks is not None:
                raise ValueError("@pllm.custom_attention needs ordinary RoPE attention without ALiBi, softcap or sinks")
            self.installed.append((impl, impl.forward))
            impl.forward = AttentionHook(self, attention, impl, layer).forward

    def prepare(self, rows: PackedPhaseRowBatch, sampler: PackedSamplerRowBatch) -> None:
        """Stage sampled-to-packed indices before model and fact-snapshot tasks run."""
        if self.program.prompt_lens.numel():
            self.program.prompt_lens[: len(rows.request_rows)].copy_(
                upload(
                    tuple(row.prompt_length for row in rows.request_rows),
                    self.program.prompt_lens.device,
                )
            )
        if self.program.sample_rows.numel():
            mapping = tuple(
                link.packed_row_index for link in sorted(sampler.links, key=lambda link: link.sampler_row_index)
            )
            self.program.sample_rows[: len(mapping)].copy_(upload(mapping, self.program.sample_rows.device))

    def close(self) -> None:
        """Restore the upstream reads before destroying the session scratch."""
        for impl, forward in reversed(self.installed):
            impl.forward = forward
        self.installed.clear()


@dataclass
class AttentionHook:
    """One layer's read at the upstream attention boundary; native cache writes precede it."""

    owner: CustomAttention
    attention: Attention
    impl: FlashAttentionImpl
    layer: int

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
    ) -> torch.Tensor:
        assert output_scale is None and output_block_scale is None
        metadata = attn_metadata
        if metadata is None:
            return output.zero_()
        if (
            VLLM_027
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
        self.owner.sites.count = count
        read = FlashRead(self.impl, metadata, output[:count])
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
        program, sites = self.owner.program, self.owner.sites
        method = program.methods[self.layer]
        kwargs = {
            name: source.value(sites, self.layer)
            if isinstance(source, (StateInput, DepthInput))
            else sites.current_rows(program.facts[source])
            for name, source in method.inputs
        }
        result = method.function(q, kv, **kwargs)
        out, aux = result if isinstance(result, tuple) else (result, None)
        output[:count].copy_(out)
        for source, tensor in ((AttentionFact(self.layer, "lse"), read.lse), (AttentionFact(self.layer, "aux"), aux)):
            if source in program.facts:
                assert tensor is not None
                sites.store_rows(program.facts[source], tensor)
        return output

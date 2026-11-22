"""Causal training for the dense vLLM decoders over borrowed weights and RoPE caches.

Quantized checkpoints train like QLoRA: each frozen projection runs its own inference kernel forward, and its
input gradient uses the dense weight vLLM dequantizes for that one backward step.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Protocol

import torch
from torch.nn import functional as F
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.phi3_long_rope_scaled_rope import Phi3LongRoPEScaledRotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.model_executor.models.gemma2 import Gemma2Attention, Gemma2DecoderLayer
from vllm.model_executor.models.gemma3 import Gemma3Attention, Gemma3DecoderLayer
from vllm.model_executor.models.glm4 import Glm4DecoderLayer
from vllm.model_executor.models.mistral import MistralAttention, MistralDecoderLayer
from vllm.model_executor.models.qwen2 import Qwen2Attention
from vllm.model_executor.models.qwen3 import Qwen3Attention

from pllm.backends.vllm.architecture import (
    DENSE_ARCHITECTURES,
    Decoder,
    DenseLayer,
    Norm,
    frozen_weight,
    lm_head_module,
)
from pllm.contracts.parameter_overlay import ParameterOverlayFieldBinding
from pllm.execution.causal_batch import LMHead
from pllm.execution.parameter_projection import parameter_correction

Project = Callable[[LinearBase, torch.Tensor], torch.Tensor]


class _ProjectionContext(Protocol):
    projection: LinearBase


class FrozenProjection(torch.autograd.Function):
    """A frozen projection exactly as inference runs it, differentiable with respect to its input only.

    No dense copy of a quantized weight is kept: backward asks vLLM to dequantize it again, and the copy
    is freed when that layer's backward step ends.
    """

    @staticmethod
    def forward(ctx: _ProjectionContext, hidden: torch.Tensor, projection: LinearBase) -> torch.Tensor:
        ctx.projection = projection
        return projection.quant_method.apply(projection, hidden, projection.bias)

    @staticmethod
    def backward(ctx: _ProjectionContext, *grads: torch.Tensor) -> tuple[torch.Tensor, None]:
        (grad,) = grads
        return grad @ frozen_weight(ctx.projection), None


def frozen_projection(hidden: torch.Tensor, projection: LinearBase) -> torch.Tensor:
    """Apply `FrozenProjection`; torch types Function.apply as returning Any."""
    return FrozenProjection.apply(hidden, projection)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def norm(module: Norm, hidden: torch.Tensor) -> torch.Tensor:
    """Normalize the dense families' full hidden width in FP32, rounding once to the model dtype."""
    weight = module.weight.float()
    if isinstance(module, GemmaRMSNorm):
        weight = weight + 1.0
    else:
        assert module.variance_size_override is None
    return F.rms_norm(hidden.float(), (hidden.shape[-1],), weight, module.variance_epsilon).to(hidden.dtype)


def add_norm(module: Norm, hidden: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused kernel stores the model-dtype sum as the residual and normalizes that rounded sum."""
    residual = hidden + residual
    return norm(module, residual), residual


def rotary(
    module: RotaryEmbedding | Phi3LongRoPEScaledRotaryEmbedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use FP32 rotary arithmetic without replacing or modifying the borrowed cache."""
    if isinstance(module, Phi3LongRoPEScaledRotaryEmbedding):
        q, k = module(positions.flatten(), query.flatten(0, 1), key.flatten(0, 1))
        return q.reshape_as(query), k.reshape_as(key)
    rotated_query, rotated_key = RotaryEmbedding.forward_static(
        positions,
        query.float(),
        key.float(),
        module.head_size,
        module.rotary_dim,
        module.cos_sin_cache,
        module.is_neox_style,
    )
    return rotated_query.to(query.dtype), rotated_key.to(key.dtype)


def attention_forward(
    layer: DenseLayer,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor | None,
    project: Project,
    qk_norms: tuple[Norm, Norm] | None,
) -> torch.Tensor:
    """Apply projection overlays before per-head normalization, RoPE and causal attention."""
    attention = layer.self_attn
    batch_size, length, _ = hidden.shape
    packed = project(attention.qkv_proj, hidden)
    query, key, value = torch.split(packed, [attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
    if qk_norms is not None:
        query = norm(qk_norms[0], query.reshape(batch_size, length, -1, attention.head_dim)).reshape_as(query)
        key = norm(qk_norms[1], key.reshape(batch_size, length, -1, attention.head_dim)).reshape_as(key)
    query, key = rotary(attention.rotary_emb, positions, query, key)
    if isinstance(attention, MistralAttention) and attention.do_llama_4_scaling:
        query = (query * attention._get_llama_4_attn_scale(positions)).to(query.dtype)
    query = query.view(batch_size, length, attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(batch_size, length, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
    value = value.view(batch_size, length, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
    cap = attention.config.attn_logit_softcapping if isinstance(attention, Gemma2Attention) else None
    scale = attention.scaling if isinstance(attention, (Gemma2Attention, Gemma3Attention)) else attention.head_dim**-0.5
    hidden = causal_attention(query, key, value, mask, attention.attn.sliding_window, cap, scale)
    hidden = hidden.transpose(1, 2).reshape(batch_size, length, attention.q_size)
    return project(attention.o_proj, hidden)


def causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    window: int | None,
    soft_cap: float | None,
    scale: float,
) -> torch.Tensor:
    """Apply the layer's score transform before masking and softmax, including fully masked padding rows."""
    if window is not None or soft_cap is not None:
        length = query.shape[2]
        visible = torch.ones((length, length), dtype=torch.bool, device=query.device).tril()
        if window is not None:
            visible = visible.triu(1 - window)
        mask = visible if mask is None else mask & visible
    if mask is not None:
        groups = query.shape[1] // key.shape[1]
        key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    if soft_cap is None:
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=mask is None, enable_gqa=True, scale=scale
        )
    assert mask is not None
    scores = (query.float() @ key.float().transpose(-1, -2)) * scale
    capped = torch.tanh(scores / soft_cap) * soft_cap
    # A fully masked row must have zero output and zero gradient.
    scores = capped.masked_fill(~mask, torch.finfo(capped.dtype).min)
    return scores.softmax(-1).masked_fill(~mask, 0).to(value.dtype) @ value


def mlp_forward(layer: DenseLayer, hidden: torch.Tensor, project: Project) -> torch.Tensor:
    """Keep fused SwiGLU projections and activation storage in the model dtype."""
    hidden = project(layer.mlp.gate_up_proj, hidden)
    hidden = layer.mlp.act_fn.forward_native(hidden)
    return project(layer.mlp.down_proj, hidden)


def feedforward(
    layer: DenseLayer, branch: torch.Tensor, residual: torch.Tensor, project: Project
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the branch norms around each family's MLP and retain its residual stream."""
    if isinstance(layer, (Gemma2DecoderLayer, Gemma3DecoderLayer)):
        hidden, residual = add_norm(
            layer.pre_feedforward_layernorm, norm(layer.post_attention_layernorm, branch), residual
        )
        return norm(layer.post_feedforward_layernorm, mlp_forward(layer, hidden, project)), residual
    if isinstance(layer, Glm4DecoderLayer):
        branch = norm(layer.post_self_attn_layernorm, branch)
    hidden, residual = add_norm(layer.post_attention_layernorm, branch, residual)
    if isinstance(layer, MistralDecoderLayer) and layer.ada_rms_norm_t_cond is not None:
        raise ValueError("causal training needs a Mistral checkpoint without a conditioning tensor")
    hidden = mlp_forward(layer, hidden, project)
    return (norm(layer.post_mlp_layernorm, hidden) if isinstance(layer, Glm4DecoderLayer) else hidden), residual


def bind_qk_norms(layer: DenseLayer) -> tuple[Norm, Norm] | None:
    """Resolve the sole attention family difference once when borrowing the model."""
    attention = layer.self_attn
    if (
        isinstance(attention, (Qwen3Attention, Gemma3Attention))
        or isinstance(attention, Qwen2Attention)
        and attention.qk_norm
    ):
        return attention.q_norm, attention.k_norm
    return None


class CausalTraining:
    """Frozen decoder arithmetic with projection views bound once by the worker."""

    def __init__(self, decoder: Decoder, bindings: tuple[ParameterOverlayFieldBinding, ...]) -> None:
        model = decoder.writable("causal training")
        if not isinstance(model, DENSE_ARCHITECTURES):
            raise ValueError(f"causal training needs a dense decoder; {type(model).__name__} uses another layer layout")
        self.model = model
        head = lm_head_module(model)
        embeddings = (model.model.embed_tokens, head)
        if (
            not all(isinstance(module.quant_method, UnquantizedEmbeddingMethod) for module in embeddings)
            or model.model.embed_tokens.weight.dtype is not torch.bfloat16
        ):
            raise ValueError("causal training needs a BF16 model whose embedding and LM head are not quantized")
        self.projections: dict[LinearBase, list[tuple[int, int, ParameterOverlayFieldBinding, slice]]] = {}
        start = 0
        for binding in bindings:
            projection, offset, size, _ = decoder.projection(binding.target)
            end = start + len(binding.fields)
            self.projections.setdefault(projection, []).append((offset, size, binding, slice(start, end)))
            start = end
        self.qk_norms = tuple(bind_qk_norms(layer) for layer in self.model.model.layers)
        self.head = LMHead(
            head.weight, model.config.vocab_size, model.logits_processor.scale, model.logits_processor.soft_cap
        )

    def project(
        self, projection: LinearBase, hidden: torch.Tensor, *, factors: tuple[torch.Tensor, ...] | None
    ) -> torch.Tensor:
        """Run the fused base projection once and correct its bound output slices."""
        output = frozen_projection(hidden, projection)
        if factors is not None:
            for offset, size, binding, fields in self.projections.get(projection, ()):
                correction = parameter_correction(hidden, factors[fields], binding.operator) * binding.scale
                # Out of place: the base output belongs to FrozenProjection's autograd node.
                corrected = output[..., offset : offset + size] + correction
                output = output.slice_scatter(corrected, dim=-1, start=offset, end=offset + size)
        return output

    def hidden(
        self, ids: torch.Tensor, attention_mask: torch.Tensor | None, factors: tuple[torch.Tensor, ...] | None
    ) -> torch.Tensor:
        """Full sequence-local causal attention with vLLM component precision and PTTS factors."""
        tokens = ids
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device).expand_as(tokens)
        mask = None
        if attention_mask is not None:
            causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).tril()
            mask = causal[None, None] & attention_mask[:, None, None, :].bool()
        hidden = self.model.model.embed_input_ids(tokens)
        project = partial(self.project, factors=factors)
        residual = None
        for index, layer in enumerate(self.model.model.layers):
            if residual is None:
                residual = hidden
                hidden = norm(layer.input_layernorm, hidden)
            else:
                hidden, residual = add_norm(layer.input_layernorm, hidden, residual)
            hidden = attention_forward(
                layer,
                hidden,
                positions,
                mask,
                project,
                self.qk_norms[index],
            )
            hidden, residual = feedforward(layer, hidden, residual, project)
        assert residual is not None
        hidden, _ = add_norm(self.model.model.norm, hidden, residual)
        return hidden

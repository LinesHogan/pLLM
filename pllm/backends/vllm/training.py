"""Causal training for the dense vLLM decoders (Qwen2, Llama, Qwen3) over borrowed weights and RoPE caches.

Quantized checkpoints train like QLoRA: each frozen projection runs its own inference kernel forward, and its
input gradient uses the dense weight vLLM dequantizes for that one backward step.
"""

from collections.abc import Callable
from functools import partial
from typing import Protocol

import torch
from torch.nn import functional as F
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
from vllm.model_executor.models.qwen2 import Qwen2Attention
from vllm.model_executor.models.qwen3 import Qwen3Attention

from pllm.backends.vllm.architecture import DENSE_ARCHITECTURES, Decoder, DenseLayer, frozen_weight
from pllm.contracts.parameter_overlay import ParameterOverlayFieldBinding
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


def norm(module: RMSNorm, hidden: torch.Tensor) -> torch.Tensor:
    """Normalize the dense families' full hidden width in FP32, rounding once to the model dtype."""
    return F.rms_norm(hidden.float(), (hidden.shape[-1],), module.weight.float(), module.variance_epsilon).to(
        hidden.dtype
    )


def add_norm(module: RMSNorm, hidden: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused kernel stores the model-dtype sum as the residual and normalizes that rounded sum."""
    residual = hidden + residual
    return norm(module, residual), residual


def rotary(
    module: RotaryEmbedding, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use FP32 rotary arithmetic without replacing or modifying the borrowed cache."""
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
    qk_norms: tuple[RMSNorm, RMSNorm] | None,
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
    query = query.view(batch_size, length, attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(batch_size, length, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
    value = value.view(batch_size, length, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
    if mask is not None:
        # Native GQA with an explicit mask falls back to the slower math backend.
        groups = attention.num_heads // attention.num_kv_heads
        key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    hidden = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, is_causal=mask is None, enable_gqa=True)
    hidden = hidden.transpose(1, 2).reshape(batch_size, length, attention.q_size)
    return project(attention.o_proj, hidden)


def mlp_forward(layer: DenseLayer, hidden: torch.Tensor, project: Project) -> torch.Tensor:
    """Keep fused SwiGLU projections and activation storage in the model dtype."""
    hidden = project(layer.mlp.gate_up_proj, hidden)
    hidden = layer.mlp.act_fn.forward_native(hidden)
    return project(layer.mlp.down_proj, hidden)


def bind_qk_norms(layer: DenseLayer) -> tuple[RMSNorm, RMSNorm] | None:
    """Resolve the sole attention family difference once when borrowing the model."""
    attention = layer.self_attn
    if isinstance(attention, Qwen3Attention) or isinstance(attention, Qwen2Attention) and attention.qk_norm:
        return attention.q_norm, attention.k_norm
    return None


class CausalTraining:
    """Frozen decoder arithmetic with projection views bound once by the worker."""

    def __init__(self, decoder: Decoder, bindings: tuple[ParameterOverlayFieldBinding, ...]) -> None:
        model = decoder.writable("causal training")
        if not isinstance(model, DENSE_ARCHITECTURES):
            raise ValueError(
                f"causal training implements dense Qwen2, Llama and Qwen3 decoders; choose one of those "
                f"instead of {type(model).__name__}"
            )
        self.model = model
        embeddings = (model.model.embed_tokens, model.lm_head)
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

    def __call__(
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
        hidden = F.embedding(tokens, self.model.model.embed_tokens.weight)
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
            hidden, residual = add_norm(layer.post_attention_layernorm, hidden, residual)
            hidden = mlp_forward(layer, hidden, project)
        assert residual is not None
        hidden, _ = add_norm(self.model.model.norm, hidden, residual)
        logits = F.linear(hidden, self.model.lm_head.weight)[:, :, : self.model.config.vocab_size]
        return logits * self.model.logits_processor.scale

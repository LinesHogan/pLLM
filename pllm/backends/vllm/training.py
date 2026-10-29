"""Causal training for the admitted vLLM decoders over borrowed weights and RoPE caches."""

from math import sqrt
from typing import NoReturn

import torch
from torch.nn import functional as F
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.models.qwen2 import Qwen2Attention
from vllm.model_executor.models.qwen3 import Qwen3Attention

from pllm.authoring import CausalAdapter, CausalTrainBatch
from pllm.backends.vllm.architecture import CausalLM, DecoderLayer
from pllm.contracts.codec import boundary_digest
from pllm.contracts.parameter_overlay import ParameterTargetShape
from pllm.contracts.tensor_layout import TensorDType


def norm(module: RMSNorm, hidden: torch.Tensor) -> torch.Tensor:
    """RMSNorm as vLLM's CUDA kernel computes it: FP32 through the weight multiply, one rounding.

    vLLM's native ATen body rounds before the weight multiply; it disagrees with the inference kernel on
    about a quarter of elements and roughly doubles training-vs-inference logit error.
    """
    value = hidden.float()
    variance = value if module.variance_size_override is None else value[..., : module.variance_size_override]
    value = value * torch.rsqrt(variance.square().mean(dim=-1, keepdim=True) + module.variance_epsilon)
    return (value * module.weight.float()).to(hidden.dtype)


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
    layer: DecoderLayer,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor | None,
    factors: tuple[torch.Tensor, ...] | None,
    scale: float,
    qk_norms: tuple[RMSNorm, RMSNorm] | None,
) -> torch.Tensor:
    """Apply Q/V corrections before per-head normalization, RoPE and causal attention."""
    attention = layer.self_attn
    batch_size, length, _ = hidden.shape
    packed = F.linear(hidden, attention.qkv_proj.weight, attention.qkv_proj.bias)
    query, key, value = torch.split(packed, [attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
    if factors is not None:
        query = query + F.linear(F.linear(hidden, factors[0]), factors[1]) * scale
        value = value + F.linear(F.linear(hidden, factors[2]), factors[3]) * scale
    if qk_norms is not None:
        query = norm(qk_norms[0], query.reshape(batch_size, length, -1, attention.head_dim)).reshape_as(query)
        key = norm(qk_norms[1], key.reshape(batch_size, length, -1, attention.head_dim)).reshape_as(key)
    query, key = rotary(attention.rotary_emb, positions, query, key)
    query = query.view(batch_size, length, attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(batch_size, length, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
    value = value.view(batch_size, length, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
    groups = attention.num_heads // attention.num_kv_heads
    key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    hidden = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, is_causal=mask is None)
    hidden = hidden.transpose(1, 2).reshape(batch_size, length, attention.q_size)
    return F.linear(hidden, attention.o_proj.weight, attention.o_proj.bias)


def mlp_forward(layer: DecoderLayer, hidden: torch.Tensor) -> torch.Tensor:
    """Keep fused SwiGLU projections and activation storage in the model dtype."""
    hidden = F.linear(hidden, layer.mlp.gate_up_proj.weight, layer.mlp.gate_up_proj.bias)
    hidden = layer.mlp.act_fn.forward_native(hidden)
    return F.linear(hidden, layer.mlp.down_proj.weight, layer.mlp.down_proj.bias)


def bind_qk_norms(layer: DecoderLayer) -> tuple[RMSNorm, RMSNorm] | None:
    """Resolve the sole attention family difference once when borrowing the model."""
    attention = layer.self_attn
    if isinstance(attention, Qwen3Attention) or isinstance(attention, Qwen2Attention) and attention.qk_norm:
        return attention.q_norm, attention.k_norm
    return None


class CausalTraining:
    """Backend causal capability; meta factors describe the driver plan without loading weights."""

    def __init__(
        self,
        adapter: CausalAdapter,
        targets: tuple[ParameterTargetShape, ...],
        tensor_dtype: TensorDType,
        seed: int,
        model_fingerprint: str,
        model: CausalLM | None = None,
    ) -> None:
        self.model = model
        self.qk_norms = () if model is None else tuple(bind_qk_norms(layer) for layer in model.model.layers)
        self.targets, self.tensor_dtype = targets, tensor_dtype
        self.scale = adapter.alpha / adapter.rank
        self.device = torch.device("meta") if model is None else model.model.embed_tokens.weight.device
        self.initial_factors = tuple(
            torch.empty(shape, device=self.device, dtype=tensor_dtype.torch)
            for target in targets
            for shape in ((adapter.rank, target.input_size), (target.output_size, adapter.rank))
        )
        if model is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            for down, up in zip(self.initial_factors[::2], self.initial_factors[1::2], strict=True):
                torch.nn.init.kaiming_uniform_(down, a=sqrt(5), generator=generator)
                up.zero_()
        self.resident_bytes = sum(t.numel() * t.element_size() for t in self.initial_factors)
        self.fingerprint = boundary_digest(
            adapter, targets, tensor_dtype, seed, "vllm.causal.training@2", model_fingerprint
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("transport CausalAdapter declarations; bind borrowed decoder weights in the worker")

    def logits(self, batch: CausalTrainBatch[torch.Tensor], factors: tuple[torch.Tensor, ...] | None) -> torch.Tensor:
        """Full sequence-local causal attention with vLLM component precision and PTTS factors."""
        assert self.model is not None
        tokens = batch.input_ids
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device).expand_as(tokens)
        mask = None
        if batch.attention_mask is not None:
            causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).tril()
            mask = causal[None, None] & batch.attention_mask[:, None, None, :].bool()
        hidden = F.embedding(tokens, self.model.model.embed_tokens.weight)
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
                None if factors is None else factors[4 * index : 4 * index + 4],
                self.scale,
                self.qk_norms[index],
            )
            hidden, residual = add_norm(layer.post_attention_layernorm, hidden, residual)
            hidden = mlp_forward(layer, hidden)
        assert residual is not None
        hidden, _ = add_norm(self.model.model.norm, hidden, residual)
        logits = F.linear(hidden, self.model.lm_head.weight)[:, :, : self.model.config.vocab_size]
        return logits * self.model.logits_processor.scale

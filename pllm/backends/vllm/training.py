"""Qwen2 causal training arithmetic over vLLM's borrowed base weights and RoPE cache."""

from math import sqrt
from typing import NoReturn

import torch
from torch.nn import functional as F
from vllm.ir.ops.layernorm import fused_add_rms_norm, rms_norm
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM

from pllm.authoring import CausalAdapter, CausalTrainBatch
from pllm.contracts.codec import boundary_digest
from pllm.contracts.parameter_overlay import ParameterTargetShape
from pllm.contracts.tensor_layout import TensorDType

# forward_native dispatches an inference IR provider; these are its autograd-capable ATen bodies.
native_norm = rms_norm.impls["native"].impl_fn
native_add_norm = fused_add_rms_norm.impls["native"].impl_fn


def add_norm(module: RMSNorm, hidden: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain the BF16 residual while normalizing the FP32 sum."""
    return native_add_norm(hidden, residual, module.weight, module.variance_epsilon, module.variance_size_override)


def layer_forward(
    layer: Qwen2DecoderLayer,
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    positions: torch.Tensor,
    mask: torch.Tensor | None,
    factors: tuple[torch.Tensor, ...] | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate the qualified fused projections and native components."""
    attention = layer.self_attn
    batch_size, length, _ = hidden.shape
    if residual is None:
        residual = hidden
        hidden = native_norm(
            hidden,
            layer.input_layernorm.weight,
            layer.input_layernorm.variance_epsilon,
            layer.input_layernorm.variance_size_override,
        )
    else:
        hidden, residual = add_norm(layer.input_layernorm, hidden, residual)
    packed = F.linear(hidden, attention.qkv_proj.weight, attention.qkv_proj.bias)
    query, key, value = torch.split(packed, [attention.q_size, attention.kv_size, attention.kv_size], dim=-1)
    if factors is not None:
        query = query + F.linear(F.linear(hidden, factors[0]), factors[1]) * scale
        value = value + F.linear(F.linear(hidden, factors[2]), factors[3]) * scale
    query, key = attention.rotary_emb.forward_native(positions, query, key)
    query = query.view(batch_size, length, attention.num_heads, attention.head_dim).transpose(1, 2)
    key = key.view(batch_size, length, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
    value = value.view(batch_size, length, attention.num_kv_heads, attention.head_dim).transpose(1, 2)
    groups = attention.num_heads // attention.num_kv_heads
    key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    hidden = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, is_causal=mask is None)
    hidden = hidden.transpose(1, 2).reshape(batch_size, length, attention.q_size)
    hidden = F.linear(hidden, attention.o_proj.weight, attention.o_proj.bias)
    hidden, residual = add_norm(layer.post_attention_layernorm, hidden, residual)
    hidden = F.linear(hidden, layer.mlp.gate_up_proj.weight, layer.mlp.gate_up_proj.bias)
    hidden = layer.mlp.act_fn.forward_native(hidden)
    return F.linear(hidden, layer.mlp.down_proj.weight, layer.mlp.down_proj.bias), residual


class Qwen2Training:
    """Backend causal capability; meta factors describe the driver plan without loading weights."""

    def __init__(
        self,
        adapter: CausalAdapter,
        targets: tuple[ParameterTargetShape, ...],
        tensor_dtype: TensorDType,
        seed: int,
        model: Qwen2ForCausalLM | None = None,
    ) -> None:
        self.model = model
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
        self.fingerprint = boundary_digest(adapter, targets, tensor_dtype, seed, "vllm.qwen2.training@1")

    def __reduce__(self) -> NoReturn:
        raise TypeError("transport CausalAdapter declarations; bind borrowed Qwen2 weights in the worker")

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
            hidden, residual = layer_forward(
                layer,
                hidden,
                residual,
                positions,
                mask,
                None if factors is None else factors[4 * index : 4 * index + 4],
                self.scale,
            )
        assert residual is not None
        hidden, _ = add_norm(self.model.model.norm, hidden, residual)
        return F.linear(hidden, self.model.lm_head.weight)[:, :, : self.model.config.vocab_size]

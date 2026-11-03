"""Shared decoder structure and native logits bound once by the vLLM runners.

All admitted families expose the same hidden sites. Block taps keep the fresh
layer output, without its separately returned residual. Dense Qwen2, Llama and
Qwen3 models support every feature, quantized or not: layer reruns call vLLM's own
quantized modules, overlays correct projection outputs, and training differentiates
through dequantized weights (see `frozen_weight`). Qwen3-MoE keeps Qwen3 attention and
KV but routes its MLP through experts; the Qwen3.5 hybrids interleave Gated
DeltaNet layers, whose recurrent state is not KV, with gated full attention.
`Decoder.projection` and `Decoder.kv_writable` decide which projections and layer
reruns those two admit. Other admitted models run read-only workloads (hidden
taps, sampler stages, activation patches).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias, overload

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.marlin import MarlinFP8ScaledMMLinearKernel
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQLinearMethod, AutoAWQMarlinLinearMethod
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (
    CompressedTensorsW8A16Fp8,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    CompressedTensorsWNA16,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import get_and_maybe_dequant_weights
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2Model,
    DeepseekV3ForCausalLM,
)
from vllm.model_executor.models.gpt_oss import GptOssForCausalLM, GptOssModel, TransformerBlock
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaModel
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM, Qwen3Model
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from vllm.scalar_type import scalar_types

from pllm.backends.vllm.compat import VLLM_027
from pllm.backends.vllm.marlin import dequantize
from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.parameter_overlay import Projection

# Importing vLLM's qwen3_next (and qwen3_5, which builds on it) initializes CUDA, which the frontend
# process of a multiprocess engine must not do; the worker-side methods below import them where used.
if TYPE_CHECKING:
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5ForCausalLM,
        Qwen3_5Model,
        Qwen3_5MoeForCausalLM,
    )

DenseLM = Qwen2ForCausalLM | LlamaForCausalLM | Qwen3ForCausalLM
# Families whose layer state lives only in KV pages, so custom forwards and PositionState can recompute it.
KvLM = DenseLM | Qwen3MoeForCausalLM
WritableLM: TypeAlias = "KvLM | Qwen3_5ForCausalLM | Qwen3_5MoeForCausalLM"
CausalLM: TypeAlias = "WritableLM | DeepseekV2ForCausalLM | GptOssForCausalLM"
DecoderBody: TypeAlias = (
    "Qwen2Model | LlamaModel | Qwen3Model | Qwen3MoeModel | Qwen3_5Model | DeepseekV2Model | GptOssModel"
)
DenseLayer = Qwen2DecoderLayer | LlamaDecoderLayer | Qwen3DecoderLayer
DecoderLayer: TypeAlias = (
    "DenseLayer | Qwen3MoeDecoderLayer | Qwen3_5DecoderLayer | DeepseekV2DecoderLayer | TransformerBlock"
)
Logits = Callable[[Tensor], Tensor | None]
Norm: TypeAlias = "RMSNorm | GemmaRMSNorm"


class NormForward(Protocol):
    """A norm's forward: normalize, or with the previous sublayer's output add it into the residual stream first."""

    @overload
    def __call__(self, x: Tensor, residual: None = None) -> Tensor: ...
    @overload
    def __call__(self, x: Tensor, residual: Tensor) -> tuple[Tensor, Tensor]: ...


LinearOutput = Tensor | tuple[Tensor, Parameter | None]
MlpForward = Callable[[Tensor], Tensor]


def frozen_weight(projection: LinearBase) -> Tensor:
    """Borrow dense storage or dequantize one transient [output, input] operand.

    Marlin A16 weights have a direct tile inverse. Other layouts keep vLLM's
    dequantizer, including its native FP8 scale application and identity fallback.
    """
    method = projection.quant_method
    if isinstance(method, AutoAWQLinearMethod):
        return ops.awq_dequantize(
            projection.get_parameter("qweight"),
            projection.get_parameter("scales"),
            projection.get_parameter("qzeros"),
            0,
            0,
            0,
        ).T.to(projection.params_dtype)
    kernel = None
    if isinstance(method, (AutoGPTQLinearMethod, AutoAWQMarlinLinearMethod)):
        kernel = method.kernel
    elif isinstance(method, (Fp8LinearMethod, Fp8PerTensorOnlineLinearMethod)):
        kernel = method.fp8_linear
    elif isinstance(method, CompressedTensorsLinearMethod):
        scheme = projection.scheme
        if isinstance(scheme, CompressedTensorsWNA16):
            kernel = scheme.kernel
        elif isinstance(scheme, CompressedTensorsW8A16Fp8):
            kernel = scheme.linear_kernel
    shape = (projection.input_size_per_partition, projection.output_size_per_partition)
    if projection.params_dtype == torch.bfloat16:
        if isinstance(kernel, MarlinLinearKernel) and kernel.config.act_type == torch.bfloat16:
            # 0.27 also admits activation-order GPTQ; its permuted K axis uses vLLM's inverse.
            if not VLLM_027 or not kernel.config.has_g_idx:
                assert kernel.w_zp_name is not None
                return dequantize(
                    projection.get_parameter(kernel.w_q_name),
                    projection.get_parameter(kernel.w_s_name),
                    projection.get_parameter(kernel.w_zp_name) if kernel.config.zero_points else None,
                    shape,
                    kernel.config.weight_type,
                    kernel.config.group_size,
                )
        elif isinstance(kernel, MarlinFP8ScaledMMLinearKernel):
            scales = projection.get_parameter("weight_scale_inv" if kernel.block_quant else "weight_scale")
            weight = projection.get_parameter("weight")
            return dequantize(
                weight,
                scales,
                None,
                shape,
                scalar_types.float8_e4m3fn,
                weight.shape[0] * 16 // scales.shape[0] if kernel.block_quant else -1,
            )
    return get_and_maybe_dequant_weights(projection, projection.params_dtype)


class Mlp(Protocol):
    """MLP call whose input supplies the down-projection training target."""

    forward: MlpForward


@dataclass(frozen=True, slots=True)
class Decoder:
    """Borrowed model and its logits callable, saved before runner hooks are installed."""

    model: CausalLM
    compute_logits: Logits
    # Why only read-only workloads run on this model; None when every workload does.
    read_only: str | None
    # The norms between sublayers (see `sublayer_norms`), as their own forwards saved before any tap wraps them.
    norms: tuple[NormForward, ...]

    def writable(self, feature: str) -> WritableLM:
        """The model for `feature`, which rewrites weights or reruns layers; read-only models reject it."""
        if self.read_only is not None:
            raise ValueError(f"{feature} {self.read_only}")
        assert not isinstance(self.model, (DeepseekV2ForCausalLM, GptOssForCausalLM))
        return self.model

    def kv_writable(self, feature: str) -> KvLM:
        """The model for `feature`, which recomputes layer state from KV pages alone."""
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM

        model = self.writable(feature)
        if reason := kv_rejection(type(model).__name__):
            raise ValueError(f"{feature} {reason}")
        assert not isinstance(model, (Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM))
        return model

    @property
    def identity(self) -> str:
        """Concrete implementation identity for backend fingerprints."""
        cls = type(self.model)
        return f"{cls.__module__}.{cls.__qualname__}"

    def projection(self, target: Projection) -> tuple[LinearBase, int, int, Mlp | None]:
        """Resolve the supported projection and its disjoint output slice; reject layouts pllm cannot slice."""
        from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
        from vllm.model_executor.models.qwen3_next import Qwen3NextAttention, Qwen3NextSparseMoeBlock

        assert target.layer is not None
        layer = self.writable("a parameter overlay").model.layers[target.layer]
        if target.name == "down_proj":
            mlp = layer.mlp
            if isinstance(mlp, (Qwen3MoeSparseMoeBlock, Qwen3NextSparseMoeBlock)):
                raise ValueError(
                    f"layer {target.layer} routes tokens to experts; MLP down overlays need a dense MLP, "
                    "so overlay the attention query or value projection instead"
                )
            return mlp.down_proj, 0, mlp.down_proj.output_size_per_partition, mlp
        if target.name in ("q_proj", "v_proj"):
            if isinstance(layer, Qwen3_5DecoderLayer) and layer.layer_type != "full_attention":
                raise ValueError(
                    f"layer {target.layer} is a Gated DeltaNet layer without q/v projections; choose a full-attention "
                    "layer (see layer_types in the model config)"
                )
            attention = layer.self_attn
            # Gated attention stores qkv_proj rows as [per-head (query, gate) pairs, key, value].
            gated = isinstance(attention, Qwen3NextAttention) and attention.attn_output_gate
            if target.name == "q_proj":
                if gated:
                    raise ValueError(
                        "gated attention interleaves each head's query rows with its output-gate rows; "
                        "overlay the value projection instead"
                    )
                return attention.qkv_proj, 0, attention.q_size, None
            if target.name == "v_proj":
                start = (2 if gated else 1) * attention.q_size + attention.kv_size
                return attention.qkv_proj, start, attention.kv_size, None
        raise ValueError("parameter overlay requires an MLP down or attention q/v projection")

    def value_rows(self, layer: int) -> tuple[LinearBase, int, int, int]:
        """A full-attention layer's value slice of its qkv projection and the width of one head."""
        module, start, width, _ = self.projection(Projection("v_proj", layer))
        return module, start, width, self.writable("a value site").model.layers[layer].self_attn.head_dim


# Reviewed invariant for these exact vLLM classes: qkv_proj splits into
# [q_size, kv_size, kv_size], or [2 * q_size, kv_size, kv_size] with the Qwen3.5 output gate. The query
# slice flows only through per-head q_norm (Qwen3 and Qwen3-MoE, or Qwen2 when enabled), RoPE and the
# attention query; it cannot write same-layer K/V. Overlay corrections enter before that split.
# These families share the decoder structure; logits dispatch remains native to each model.
DENSE_ARCHITECTURES = (Qwen2ForCausalLM, LlamaForCausalLM, Qwen3ForCausalLM)
_MLA_MOE = "has MLA attention (no separate V projection, a latent KV cache) and per-expert MoE projections"
# Their decoder layers return (fresh block output, residual) and end in model.norm like the dense
# families, so hidden sites and activation patches mean the same; their projections and KV do not.
READ_ONLY_ARCHITECTURES: dict[type[CausalLM], str] = {
    DeepseekV2ForCausalLM: _MLA_MOE,
    DeepseekV3ForCausalLM: _MLA_MOE,
    GptOssForCausalLM: "has sink attention with sliding windows and per-expert MXFP4 MoE projections",
}


_BASE_ARCHITECTURES = (*DENSE_ARCHITECTURES, Qwen3MoeForCausalLM, *READ_ONLY_ARCHITECTURES)
_HYBRID_NAMES = ("Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM")


def _architectures() -> tuple[type[CausalLM], ...]:
    # The hybrid imports initialize CUDA; only workers need the concrete classes.
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM

    return (*_BASE_ARCHITECTURES, Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM)


def admit_architecture(name: str) -> str | None:
    """Admit a registry-inspected class name without importing CUDA-initializing hybrid modules."""
    if name not in (*_HYBRID_NAMES, *(cls.__name__ for cls in _BASE_ARCHITECTURES)):
        raise ValueError(
            "choose a Qwen2, Llama, Qwen3, Qwen3Moe, Qwen3_5, Qwen3_5Moe, DeepseekV2, DeepseekV3 or GptOss "
            "ForCausalLM model for vLLM workloads"
        )
    family = next((reason for cls, reason in READ_ONLY_ARCHITECTURES.items() if cls.__name__ == name), None)
    return (
        None
        if family is None
        else f"needs a Qwen2, Llama, Qwen3, Qwen3-MoE or Qwen3.5 model ({name} {family}); "
        "hidden taps, ESamp, sampler stages and activation patches still run on this model"
    )


def kv_rejection(name: str) -> str | None:
    """Why a model cannot replay its layer state from KV pages alone."""
    return (
        f"recomputes layer state from KV pages, but {name} keeps "
        "Gated DeltaNet recurrent state outside them; use a Qwen2, Llama, Qwen3 or Qwen3-MoE model"
        if name in _HYBRID_NAMES
        else None
    )


def bind_model(model: nn.Module) -> Decoder:
    """Admit a concrete vLLM architecture at model load, before any instance wrapping."""
    read_only = admit_architecture(type(model).__name__)
    architectures = _architectures()
    if type(model) not in architectures:
        raise ValueError("choose a native vLLM model class; workload admission does not accept model subclasses")
    assert isinstance(model, architectures)
    return Decoder(model, model.compute_logits, read_only, tuple(norm.forward for norm in sublayer_norms(model)))


def sublayer_norms(model: CausalLM) -> tuple[Norm, ...]:
    """The 2L + 1 norms between sublayers, in forward order: each layer's input and post-attention norm, then the
    final norm. Every admitted family calls them with (branch output, residual) and returns (normalized, sum), so
    they are where each sublayer's residual add happens; the attribute names are shared, so this is no table."""
    body = model.model
    return (
        *(norm for layer in body.layers for norm in (layer.input_layernorm, layer.post_attention_layernorm)),
        body.norm,
    )


def hidden_binding(layer: int, site: HiddenSite, layer_count: int) -> PackedHiddenResourceBinding:
    """Resolve the shared decoder sites to the module a tap wraps.

    Every admitted family computes `h, r = input_layernorm(h, r)`, attention, `h, r = post_attention_layernorm(h,
    r)`, MLP, and ends with `model.norm(h, r)`; each norm adds the previous sublayer's output into the residual
    stream and returns the sum second. So the stream after layer l is the second output of layer l+1's input norm
    (of model.norm after the last layer), and the stream between attention and MLP that of the post-attention norm.
    Layer logits are computed from residual taps by the session, at the LM head.
    """
    if layer == -1 and site in (HiddenSite.MODEL_OUTPUT, HiddenSite.LAYER_LOGITS):
        return PackedHiddenResourceBinding(
            str(layer), HiddenLayerKind.INDEX, site, "model.norm" if site is HiddenSite.MODEL_OUTPUT else "lm_head"
        )
    index = layer if layer >= 0 else layer_count + layer
    if not 0 <= index < layer_count:
        raise ValueError(f"hidden layer {layer} at site {site.value} is outside 0..{layer_count - 1}")
    match site:
        case HiddenSite.BLOCK_OUTPUT:
            path, output = f"model.layers.{index}", 0
        case HiddenSite.RESIDUAL:
            path, output = "model.norm" if index == layer_count - 1 else f"model.layers.{index + 1}.input_layernorm", 1
        case HiddenSite.ATTENTION_RESIDUAL:
            path, output = f"model.layers.{index}.post_attention_layernorm", 1
        case _:
            raise ValueError(f"hidden site {site.value} reads layer -1 only")
    return PackedHiddenResourceBinding(str(layer), HiddenLayerKind.INDEX, site, path, output)

"""Shared decoder structure and native logits bound once by the vLLM runners.

All admitted families expose the same hidden sites. Block taps keep the fresh
layer output, without its separately returned residual. Dense Qwen2, Llama and
Qwen3 models support every feature, quantized or not: layer reruns call vLLM's own
quantized modules, overlays correct projection outputs, and training differentiates
through dequantized weights (see `frozen_weight`). Qwen3-MoE keeps Qwen3 attention and
KV but routes its MLP through experts; the Qwen3.5 hybrids interleave Gated
DeltaNet layers, whose recurrent state is not KV, with gated full attention.
`Decoder.projection` and `Decoder.kv_writable` decide which projections and layer
replays those two admit. Other admitted models run read-only workloads (hidden
taps, sampler stages, activation patches).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.marlin import MarlinFP8ScaledMMLinearKernel
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
from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget

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
# Families whose layer state lives only in KV pages, so layer routes and PositionState can replay it.
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

    def writable(self, feature: str) -> WritableLM:
        """The model for `feature`, which rewrites weights or reruns layers; read-only models reject it."""
        if self.read_only is not None:
            raise ValueError(f"{feature} {self.read_only}")
        assert not isinstance(self.model, (DeepseekV2ForCausalLM, GptOssForCausalLM))
        return self.model

    def kv_writable(self, feature: str) -> KvLM:
        """The model for `feature`, which replays layer state through KV pages alone."""
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM

        model = self.writable(feature)
        if isinstance(model, (Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM)):
            raise ValueError(
                f"{feature} replays layer state through KV pages, but {type(model).__name__} keeps "
                "Gated DeltaNet recurrent state outside them; use a Qwen2, Llama, Qwen3 or Qwen3-MoE model"
            )
        return model

    @property
    def identity(self) -> str:
        """Concrete implementation identity for backend fingerprints."""
        cls = type(self.model)
        return f"{cls.__module__}.{cls.__qualname__}"

    def projection(self, target: ParameterTarget) -> tuple[LinearBase, int, int, Mlp | None]:
        """Resolve the supported projection and its disjoint output slice; reject layouts pllm cannot slice."""
        from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
        from vllm.model_executor.models.qwen3_next import Qwen3NextAttention, Qwen3NextSparseMoeBlock

        assert target.layer is not None
        layer = self.writable("a parameter overlay").model.layers[target.layer]
        if target.module is ParameterModuleKind.MLP and target.role is ParameterRole.DOWN_PROJECTION_WEIGHT:
            mlp = layer.mlp
            if isinstance(mlp, (Qwen3MoeSparseMoeBlock, Qwen3NextSparseMoeBlock)):
                raise ValueError(
                    f"layer {target.layer} routes tokens to experts; MLP down overlays need a dense MLP, "
                    "so overlay the attention query or value projection instead"
                )
            return mlp.down_proj, 0, mlp.down_proj.output_size_per_partition, mlp
        if target.module is ParameterModuleKind.ATTENTION:
            if isinstance(layer, Qwen3_5DecoderLayer) and layer.layer_type != "full_attention":
                raise ValueError(
                    f"layer {target.layer} is a Gated DeltaNet layer; overlay a full-attention layer "
                    "(see layer_types in the model config)"
                )
            attention = layer.self_attn
            # Gated attention stores qkv_proj rows as [per-head (query, gate) pairs, key, value].
            gated = isinstance(attention, Qwen3NextAttention) and attention.attn_output_gate
            if target.role is ParameterRole.QUERY_PROJECTION_WEIGHT:
                if gated:
                    raise ValueError(
                        "gated attention interleaves each head's query rows with its output-gate rows; "
                        "overlay the value projection instead"
                    )
                return attention.qkv_proj, 0, attention.q_size, None
            if target.role is ParameterRole.VALUE_PROJECTION_WEIGHT:
                start = (2 if gated else 1) * attention.q_size + attention.kv_size
                return attention.qkv_proj, start, attention.kv_size, None
        raise ValueError("parameter overlay requires an MLP down or attention q/v projection")


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


def bind_model(model: nn.Module) -> Decoder:
    """Admit a concrete vLLM architecture at model load, before any instance wrapping."""
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM

    architectures = (
        *DENSE_ARCHITECTURES,
        Qwen3MoeForCausalLM,
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
        *READ_ONLY_ARCHITECTURES,
    )
    # Exact classes: vLLM subclasses such as Phi3 or TeleFLM change weights or logits scaling.
    if type(model) not in architectures or not isinstance(model, architectures):
        raise ValueError(
            "choose a Qwen2, Llama, Qwen3, Qwen3Moe, Qwen3_5, Qwen3_5Moe, DeepseekV2, DeepseekV3 or GptOss "
            "ForCausalLM model for vLLM workloads"
        )
    family = READ_ONLY_ARCHITECTURES.get(type(model))
    if family is None:
        return Decoder(model, model.compute_logits, None)
    read_only = (
        f"needs a Qwen2, Llama, Qwen3, Qwen3-MoE or Qwen3.5 model ({type(model).__name__} {family}); "
        "hidden taps, ESamp, sampler stages and activation patches still run on this model"
    )
    return Decoder(model, model.compute_logits, read_only)


def hidden_binding(layer: int, site: HiddenSite, layer_count: int) -> PackedHiddenResourceBinding:
    """Resolve the shared decoder sites: fresh block output or final normalized output."""
    if site is HiddenSite.MODEL_OUTPUT and layer == -1:
        return PackedHiddenResourceBinding(str(layer), HiddenLayerKind.INDEX, site, "model.norm")
    module_index = layer if layer >= 0 else layer_count + layer
    if site is not HiddenSite.BLOCK_OUTPUT or not 0 <= module_index < layer_count:
        raise ValueError(f"hidden layer {layer} at site {site.value} is outside 0..{layer_count - 1}")
    return PackedHiddenResourceBinding(
        str(layer), HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, f"model.layers.{module_index}"
    )

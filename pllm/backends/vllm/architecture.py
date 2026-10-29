"""Shared decoder structure and native logits bound once by the vLLM runners.

All admitted families expose the same hidden sites. Block taps keep the fresh
layer output, without its separately returned residual.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor, nn
from torch.nn import Parameter
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaModel
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM, Qwen3Model

from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget

CausalLM = Qwen2ForCausalLM | LlamaForCausalLM | Qwen3ForCausalLM
DecoderBody = Qwen2Model | LlamaModel | Qwen3Model
DecoderLayer = Qwen2DecoderLayer | LlamaDecoderLayer | Qwen3DecoderLayer
Logits = Callable[[Tensor], Tensor | None]
LinearOutput = Tensor | tuple[Tensor, Parameter | None]
LinearForward = Callable[[Tensor], LinearOutput]
MlpForward = Callable[[Tensor], Tensor]


class Projection(Protocol):
    """Weight geometry and replaceable forward shared by the projection readers."""

    weight: Parameter
    bias: Parameter | None
    tp_size: int
    return_bias: bool
    forward: LinearForward


class Mlp(Protocol):
    """MLP call whose input supplies the down-projection training target."""

    forward: MlpForward


@dataclass(frozen=True, slots=True)
class Decoder:
    """Borrowed model and its logits callable, saved before runner hooks are installed."""

    model: CausalLM
    compute_logits: Logits

    @property
    def identity(self) -> str:
        """Concrete implementation identity for backend fingerprints."""
        cls = type(self.model)
        return f"{cls.__module__}.{cls.__qualname__}"

    def projection(self, target: ParameterTarget) -> tuple[Projection, int, int, Mlp | None]:
        """Resolve the supported projection and its disjoint output slice."""
        assert target.layer is not None
        layer = self.model.model.layers[target.layer]
        if target.module is ParameterModuleKind.MLP and target.role is ParameterRole.DOWN_PROJECTION_WEIGHT:
            module = layer.mlp.down_proj
            return module, 0, int(module.weight.shape[0]), layer.mlp
        if target.module is ParameterModuleKind.ATTENTION:
            attention = layer.self_attn
            if target.role is ParameterRole.QUERY_PROJECTION_WEIGHT:
                return attention.qkv_proj, 0, attention.q_size, None
            if target.role is ParameterRole.VALUE_PROJECTION_WEIGHT:
                return attention.qkv_proj, attention.q_size + attention.kv_size, attention.kv_size, None
        raise ValueError("parameter overlay requires an MLP down or attention q/v projection")


# Reviewed invariant for these exact vLLM classes: qkv_proj splits into
# [q_size, kv_size, kv_size]. The query slice flows only through per-head q_norm
# (Qwen3, or Qwen2 when enabled), RoPE and the attention query; it cannot write
# same-layer K/V. Overlay corrections enter before that split.
# These families share the decoder structure; logits dispatch remains native to each model.
ARCHITECTURES = (Qwen2ForCausalLM, LlamaForCausalLM, Qwen3ForCausalLM)


def bind_model(model: nn.Module) -> Decoder:
    """Admit a concrete vLLM architecture at model load, before any instance wrapping."""
    # Exact classes: vLLM subclasses such as Phi3 or TeleFLM change weights or logits scaling.
    if type(model) not in ARCHITECTURES or not isinstance(model, ARCHITECTURES):
        raise ValueError("choose a Qwen2ForCausalLM, LlamaForCausalLM or Qwen3ForCausalLM model for vLLM workloads")
    if model.model.quant_config is not None:
        raise ValueError("vLLM workloads need an unquantized model")
    return Decoder(model, model.compute_logits)


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

"""Shared decoder structure and native logits bound once by the vLLM runners.

All admitted families expose the same hidden sites. Block taps keep the fresh
layer output, without its separately returned residual.
"""

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor, nn
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding

CausalLM = Qwen2ForCausalLM | LlamaForCausalLM | Qwen3ForCausalLM
Logits = Callable[[Tensor], Tensor | None]


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

    def effect_model(self) -> Qwen2ForCausalLM:
        """Qualify effects separately until their architecture-specific paths are supported."""
        if not isinstance(self.model, Qwen2ForCausalLM):
            raise ValueError(
                "parameter overlays, activation patches, replay and causal training still need a Qwen2 model"
            )
        return self.model


# These families share the decoder structure; logits dispatch remains native to each model.
ARCHITECTURES = {
    Qwen2ForCausalLM: Decoder,
    LlamaForCausalLM: Decoder,
    Qwen3ForCausalLM: Decoder,
}


def bind_model(model: nn.Module) -> Decoder:
    """Admit a concrete vLLM architecture at model load, before any instance wrapping."""
    if not isinstance(model, tuple(ARCHITECTURES)) or type(model) not in ARCHITECTURES:
        raise ValueError("choose a Qwen2ForCausalLM, LlamaForCausalLM or Qwen3ForCausalLM model for vLLM workloads")
    if model.model.quant_config is not None:
        raise ValueError("vLLM workloads need an unquantized model")
    return ARCHITECTURES[type(model)](model, model.compute_logits)


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

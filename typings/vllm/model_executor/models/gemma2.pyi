"""vLLM Gemma 2 decoder surface."""

from torch import Tensor
from torch.nn import Module
from transformers import Gemma2Config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models._decoder import DecoderAttention, DecoderLayer, DecoderMLP, DecoderModel


class Gemma2Attention(DecoderAttention):
    scaling: float
    config: Gemma2Config


class Gemma2MLP(DecoderMLP): ...


class Gemma2DecoderLayer(DecoderLayer[Gemma2Attention, Gemma2MLP]):
    input_layernorm: GemmaRMSNorm
    post_attention_layernorm: GemmaRMSNorm
    pre_feedforward_layernorm: GemmaRMSNorm
    post_feedforward_layernorm: GemmaRMSNorm


class Gemma2Model(DecoderModel[Gemma2DecoderLayer]):
    norm: GemmaRMSNorm
    normalizer: Tensor


class Gemma2ForCausalLM(Module):
    model: Gemma2Model
    logits_processor: LogitsProcessor
    config: Gemma2Config

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...

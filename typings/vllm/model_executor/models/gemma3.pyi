"""vLLM Gemma 3 decoder surface."""

from torch import Tensor
from torch.nn import Module
from transformers import Gemma3TextConfig
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models._decoder import DecoderAttention, DecoderLayer, DecoderMLP, DecoderModel


class Gemma3Attention(DecoderAttention):
    scaling: float
    q_norm: GemmaRMSNorm
    k_norm: GemmaRMSNorm


class Gemma3MLP(DecoderMLP): ...


class Gemma3DecoderLayer(DecoderLayer[Gemma3Attention, Gemma3MLP]):
    input_layernorm: GemmaRMSNorm
    post_attention_layernorm: GemmaRMSNorm
    pre_feedforward_layernorm: GemmaRMSNorm
    post_feedforward_layernorm: GemmaRMSNorm


class Gemma3Model(DecoderModel[Gemma3DecoderLayer]):
    norm: GemmaRMSNorm
    normalizer: Tensor


class Gemma3ForCausalLM(Module):
    model: Gemma3Model
    logits_processor: LogitsProcessor
    lm_head: VocabParallelEmbedding
    config: Gemma3TextConfig

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...

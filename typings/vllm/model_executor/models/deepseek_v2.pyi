"""vLLM DeepSeek V2/V3 (MLA attention, MoE) surface admitted for read-only and MLA workloads."""

from collections.abc import Callable, Sequence

from torch import Tensor
from torch.nn import Module
from transformers import DeepseekV2Config
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models._decoder import DecoderAttention


class MultiHeadLatentAttentionWrapper(Module):
    mla_attn: MLAAttention


class DeepseekV2MLAAttention(Module):
    mla_attn: MultiHeadLatentAttentionWrapper


class DeepseekV2DecoderLayer(Module):
    self_attn: DeepseekV2MLAAttention | DecoderAttention
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    forward: Callable[[Tensor, Tensor, Tensor | None, Tensor | None], tuple[Tensor, Tensor]]


class DeepseekV2Model(Module):
    layers: Sequence[DeepseekV2DecoderLayer]
    norm: RMSNorm
    do_not_compile: bool


class DeepseekV2ForCausalLM(Module):
    model: DeepseekV2Model
    lm_head: VocabParallelEmbedding
    logits_processor: LogitsProcessor
    config: DeepseekV2Config

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM): ...

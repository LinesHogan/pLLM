"""vLLM Mixtral MoE decoder surface."""

from collections.abc import Callable

from torch import Tensor
from torch.nn import Module
from transformers import MixtralConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models._decoder import DecoderModel
from vllm.model_executor.models.llama import LlamaAttention


class MixtralMoE(Module):
    def forward(self, hidden_states: Tensor) -> Tensor: ...


class MixtralDecoderLayer(Module):
    self_attn: LlamaAttention
    block_sparse_moe: MixtralMoE
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    forward: Callable[[Tensor, Tensor, Tensor | None], tuple[Tensor, Tensor]]


class MixtralModel(DecoderModel[MixtralDecoderLayer]): ...


class MixtralForCausalLM(Module):
    model: MixtralModel
    logits_processor: LogitsProcessor
    lm_head: VocabParallelEmbedding
    config: MixtralConfig

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...

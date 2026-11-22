"""vLLM Qwen2-MoE decoder surface."""

from torch import Tensor
from torch.nn import Module
from transformers import Qwen2MoeConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models._decoder import DecoderLayer, DecoderMLP, DecoderModel
from vllm.model_executor.models.llama import LlamaAttention


class Qwen2MoeMLP(DecoderMLP): ...


class Qwen2MoeSparseMoeBlock(Module):
    def forward(self, hidden_states: Tensor) -> Tensor: ...


class Qwen2MoeDecoderLayer(DecoderLayer[LlamaAttention, Qwen2MoeMLP | Qwen2MoeSparseMoeBlock]): ...


class Qwen2MoeModel(DecoderModel[Qwen2MoeDecoderLayer]): ...


class Qwen2MoeForCausalLM(Module):
    model: Qwen2MoeModel
    logits_processor: LogitsProcessor
    lm_head: VocabParallelEmbedding
    config: Qwen2MoeConfig

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...

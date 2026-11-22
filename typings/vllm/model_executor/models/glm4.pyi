"""vLLM GLM-4 decoder surface."""

from torch import Tensor
from torch.nn import Module
from transformers import Glm4Config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models._decoder import DecoderLayer, DecoderModel
from vllm.model_executor.models.llama import LlamaAttention, LlamaMLP


class Glm4DecoderLayer(DecoderLayer[LlamaAttention, LlamaMLP]):
    post_self_attn_layernorm: RMSNorm
    post_mlp_layernorm: RMSNorm


class Glm4Model(DecoderModel[Glm4DecoderLayer]): ...


class Glm4ForCausalLM(Module):
    model: Glm4Model
    logits_processor: LogitsProcessor
    lm_head: VocabParallelEmbedding
    config: Glm4Config

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...

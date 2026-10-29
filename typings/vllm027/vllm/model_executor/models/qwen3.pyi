"""vLLM Qwen3 decoder surface used by workload binding and qualified effects."""

from torch import Tensor
from torch.nn import Module
from transformers import Qwen3Config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models._decoder import DecoderAttention, DecoderLayer
from vllm.model_executor.models.qwen2 import Qwen2MLP, Qwen2Model

class Qwen3Attention(DecoderAttention):
    q_norm: RMSNorm
    k_norm: RMSNorm

class Qwen3DecoderLayer(DecoderLayer[Qwen3Attention, Qwen2MLP]): ...
class Qwen3Model(Qwen2Model[Qwen3DecoderLayer]): ...

class Qwen3ForCausalLM(Module):
    model: Qwen3Model
    logits_processor: LogitsProcessor
    lm_head: VocabParallelEmbedding
    config: Qwen3Config
    quant_config: object | None

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...

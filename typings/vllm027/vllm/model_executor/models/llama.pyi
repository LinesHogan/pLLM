"""vLLM Llama decoder surface used by workload binding and qualified effects."""

from torch import Tensor
from torch.nn import Module
from transformers import LlamaConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models._decoder import DecoderAttention, DecoderLayer, DecoderMLP, DecoderModel

class LlamaAttention(DecoderAttention): ...
class LlamaMLP(DecoderMLP): ...
class LlamaDecoderLayer(DecoderLayer[LlamaAttention, LlamaMLP]): ...
class LlamaModel(DecoderModel[LlamaDecoderLayer]): ...

class LlamaForCausalLM(Module):
    model: LlamaModel
    logits_processor: LogitsProcessor
    lm_head: VocabParallelEmbedding
    config: LlamaConfig

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...

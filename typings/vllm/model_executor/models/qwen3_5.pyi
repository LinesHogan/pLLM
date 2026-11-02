"""vLLM Qwen3.5 hybrid decoder surface: Gated DeltaNet layers interleaved with gated full attention."""

from collections.abc import Callable, Sequence

from torch import Tensor
from torch.nn import Module
from transformers import Qwen3_5MoeTextConfig, Qwen3_5TextConfig
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention, Qwen3NextMLP, Qwen3NextSparseMoeBlock
from vllm.sequence import IntermediateTensors

class Qwen3_5DecoderLayer(Module):
    # "linear_attention" layers own linear_attn instead of self_attn.
    layer_type: str
    self_attn: Qwen3NextAttention
    mlp: Qwen3NextMLP | Qwen3NextSparseMoeBlock
    input_layernorm: GemmaRMSNorm
    post_attention_layernorm: GemmaRMSNorm
    # Called with keywords: (hidden_states, residual, positions) in declaration order.
    forward: Callable[..., tuple[Tensor, Tensor]]

class Qwen3_5Model(Module):
    embed_tokens: VocabParallelEmbedding
    layers: Sequence[Qwen3_5DecoderLayer]
    norm: GemmaRMSNorm
    do_not_compile: bool
    forward: Callable[[Tensor | None, Tensor, IntermediateTensors | None, Tensor | None], Tensor | IntermediateTensors]

    def embed_input_ids(self, input_ids: Tensor) -> Tensor: ...

class Qwen3_5ForCausalLM(Module):
    model: Qwen3_5Model
    logits_processor: LogitsProcessor
    lm_head: VocabParallelEmbedding
    config: Qwen3_5TextConfig

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...

class Qwen3_5MoeForCausalLM(Module):
    model: Qwen3_5Model
    logits_processor: LogitsProcessor
    lm_head: VocabParallelEmbedding
    config: Qwen3_5MoeTextConfig

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...

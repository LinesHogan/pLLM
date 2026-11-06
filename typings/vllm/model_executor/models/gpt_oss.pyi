"""vLLM gpt-oss (sink attention, MXFP4 MoE) surface admitted for read-only workloads."""

from collections.abc import Callable, Sequence

from torch import Tensor
from torch.nn import Module
from transformers import GptOssConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

class TransformerBlock(Module):
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    # Unlike the other decoders, the block takes (hidden_states, positions, residual).
    forward: Callable[[Tensor, Tensor, Tensor | None], tuple[Tensor, Tensor]]

class GptOssModel(Module):
    layers: Sequence[TransformerBlock]
    norm: RMSNorm
    do_not_compile: bool

class GptOssForCausalLM(Module):
    model: GptOssModel
    lm_head: VocabParallelEmbedding
    logits_processor: LogitsProcessor
    config: GptOssConfig

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...

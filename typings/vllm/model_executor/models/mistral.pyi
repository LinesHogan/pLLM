"""vLLM Mistral decoder surface."""

from torch import Tensor
from torch.nn import Module
from transformers import MistralConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models._decoder import DecoderAttention, DecoderLayer, DecoderMLP, DecoderModel
from vllm.model_executor.models.llama import LlamaForCausalLM


class MistralAttention(DecoderAttention):
    do_llama_4_scaling: bool

    def _get_llama_4_attn_scale(self, positions: Tensor) -> Tensor: ...


class MistralMLP(DecoderMLP): ...


class MistralDecoderLayer(DecoderLayer[MistralAttention, MistralMLP]):
    ada_rms_norm_t_cond: Module | None


class MistralModel(DecoderModel[MistralDecoderLayer]): ...


class MistralForCausalLM(LlamaForCausalLM):
    model: MistralModel  # pyright: ignore[reportIncompatibleVariableOverride]
    logits_processor: LogitsProcessor
    lm_head: VocabParallelEmbedding
    config: MistralConfig  # pyright: ignore[reportIncompatibleVariableOverride]

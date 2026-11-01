"""vLLM Qwen3-Next components reused by the Qwen3.5 hybrid decoders."""

from torch import Tensor
from torch.nn import Module
from vllm.model_executor.models._decoder import DecoderAttention, DecoderMLP

class Qwen3NextAttention(DecoderAttention):
    # With the gate, qkv_proj rows are [per-head (query, gate) pairs, key, value].
    attn_output_gate: bool

class Qwen3NextMLP(DecoderMLP): ...

class Qwen3NextSparseMoeBlock(Module):
    def forward(self, hidden_states: Tensor) -> Tensor: ...

"""vLLM's per-method dequantization of a linear layer's weight."""

import torch
from vllm.model_executor.layers.linear import LinearBase

def get_and_maybe_dequant_weights(layer: LinearBase, out_dtype: torch.dtype = ...) -> torch.Tensor: ...

import torch
from torch.nn import Module

class Attention(Module):
    impl: object
    kv_cache: torch.Tensor
    kv_cache_dtype: str
    kv_sharing_target_layer_name: str | None
    num_heads: int
    calculate_kv_scales: bool
    query_quant: Module | None
    use_direct_call: bool
    num_kv_heads: int
    head_size: int
    layer_name: str

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheSpec
import torch
from torch.nn import Module

class Attention(Module):
    impl: object
    kv_cache: torch.Tensor
    kv_cache_dtype: str
    kv_cache_torch_dtype: torch.dtype
    sliding_window: int | None
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None: ...
    kv_sharing_target_layer_name: str | None
    num_heads: int
    calculate_kv_scales: bool
    query_quant: Module | None
    use_direct_call: bool
    num_kv_heads: int
    head_size: int
    layer_name: str

from contextlib import AbstractContextManager
from dataclasses import dataclass
import torch
from vllm.config import CUDAGraphMode

@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0

@dataclass
class ForwardContext:
    no_compile_layers: dict[str, object]
    attn_metadata: dict[str, object] | list[dict[str, object]]
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
    dp_metadata: object | None = ...
    cudagraph_runtime_mode: CUDAGraphMode = ...
    batch_descriptor: BatchDescriptor | None = ...
    ubatch_slices: object | None = ...
    is_padding: torch.Tensor | None = ...

def get_forward_context() -> ForwardContext: ...
def override_forward_context(forward_context: ForwardContext | None) -> AbstractContextManager[None]: ...

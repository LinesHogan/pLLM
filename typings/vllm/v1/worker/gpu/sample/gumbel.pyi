"""Device helper shared with the bounded post-filter draw on both supported releases."""

import torch
import triton.language as tl

def gumbel_block_argmax(*args: object, **kwargs: object) -> tuple[tl.tensor, tl.tensor]: ...
def gumbel_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor: ...

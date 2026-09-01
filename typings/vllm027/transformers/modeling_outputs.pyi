from __future__ import annotations

import torch


class CausalLMOutputWithPast:
    logits: torch.Tensor | None

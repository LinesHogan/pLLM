#!/usr/bin/env python3
"""OverRIDE iterative reweighting expressed as a tLLM Program.

The whole tensor algorithm: ``num_iterations - 1`` low-rank heads
(hidden -> vocab), stored as stacked parameters so a flow selects its head
with ``index_select``. ``reweight`` subtracts the previous iteration's
residual from the logits (read-only, via ``ProgramConsumer.run_flow``);
``forward`` trains the current iteration's head with cross-entropy against
the sampled tokens. Row grouping by iteration and the sampler-boundary
stash are plain Python in ``consumer.py``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from tllm.ports.base import TimingDeadline
from tllm.workgraph.frontend import extern, flow, local, provided, result


class OverRIDEProgram(torch.nn.Module):
    """Stacked low-rank reweighting heads trained on sampled tokens."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        num_heads: int,
        lr: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if int(num_heads) <= 0:
            raise ValueError("OverRIDEProgram requires at least one head")
        down = torch.empty((int(num_heads), int(rank), int(hidden_size)), device=device, dtype=dtype)
        nn.init.normal_(down, mean=0.0, std=1.0 / math.sqrt(float(hidden_size)))
        self.down = nn.Parameter(down)
        self.up = nn.Parameter(
            torch.zeros((int(num_heads), int(vocab_size), int(rank)), device=device, dtype=dtype)
        )
        self.optimizer = torch.optim.Adam((self.down, self.up), lr=lr, foreach=False)

    def _head_logits(self, hidden: torch.Tensor, head_idx: torch.Tensor) -> torch.Tensor:
        idx = head_idx.reshape(-1)
        down = self.down.index_select(0, idx)  # (rows, rank, hidden)
        up = self.up.index_select(0, idx)  # (rows, vocab, rank)
        out = torch.bmm(torch.bmm(hidden.unsqueeze(1), down.transpose(1, 2)), up.transpose(1, 2))
        return out.squeeze(1)

    @flow(
        inputs={
            "hidden": provided(),
            "logits": provided(),
            "head_idx": provided(),
            "scale": provided(),
        },
        outputs={"reweighted": local()},
        state="heads",
    )
    def reweight(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        head_idx: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        # Rows without a previous-iteration head pass scale=0 (index is
        # clamped to 0 and multiplied away).
        residual = self._head_logits(hidden, head_idx).to(dtype=logits.dtype)
        return logits - scale.reshape(-1, 1).to(dtype=logits.dtype) * residual

    @flow(
        inputs={
            "hidden": provided(),
            "logits": provided(),
            "head_idx": provided(),
            "tokens": provided(),
        },
        outputs={"loss": local()},
        state="heads",
    )
    def forward(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        head_idx: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        q_logits = logits + self._head_logits(hidden, head_idx).to(dtype=logits.dtype)
        return F.cross_entropy(q_logits, tokens.reshape(-1).to(dtype=torch.long))

    @extern(
        inputs={"loss": result("forward", "loss")},
        after=("forward",),
        state="heads",
        deadline=TimingDeadline.NEXT_PREPARE,
    )
    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()


__all__ = ["OverRIDEProgram"]

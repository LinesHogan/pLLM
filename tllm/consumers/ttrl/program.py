#!/usr/bin/env python3
"""TTRL reward model expressed as a tLLM Program.

The whole tensor algorithm: a low-rank adapter mapping hidden rows to a
scalar reward prediction, trained with MSE against majority-vote rewards.
The verifier, voting, and retry/barrier orchestration are plain Python in
``rewards.py``; everything about *executing* this training (lanes,
functional optimizer transitions, CUDA graph capture) is derived by
``tllm.program.compile_program``.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from tllm.consumers._common.alignment import LowRankAdapter
from tllm.ports.base import TimingDeadline
from tllm.workgraph.frontend import extern, flow, local, provided, result


class TTRLProgram(torch.nn.Module):
    """Predict the majority-vote reward of a row from its hidden state."""

    def __init__(
        self,
        *,
        hidden_size: int,
        rank: int,
        lr: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.adapter = LowRankAdapter(
            in_dim=hidden_size, out_dim=1, rank=rank, dtype=dtype, device=device
        )
        self.optimizer = torch.optim.Adam(
            self.adapter.parameters(), lr=lr, foreach=False
        )

    @flow(
        inputs={"hidden": provided(), "rewards": provided()},
        outputs={"loss": local()},
        state="reward_model",
    )
    def forward(self, hidden: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        target = rewards.reshape(-1, 1).to(device=hidden.device, dtype=hidden.dtype)
        return F.mse_loss(self.adapter(hidden), target)

    @flow(
        inputs={"hidden": provided()},
        outputs={"score": local()},
        state="reward_model",
    )
    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        # Read-only invocation (ProgramConsumer.run_flow); the PG300
        # "unconsumed local" diagnostic is expected.
        return self.adapter(hidden)

    @extern(
        inputs={"loss": result("forward", "loss")},
        after=("forward",),
        state="reward_model",
        deadline=TimingDeadline.NEXT_PREPARE,
    )
    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()


__all__ = ["TTRLProgram"]

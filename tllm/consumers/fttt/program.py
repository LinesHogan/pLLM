#!/usr/bin/env python3
"""FTTT feedback-edit adapter expressed as a tLLM Program.

The whole tensor algorithm: a low-rank adapter mapping an edit input
(hidden row + hashed feedback features + reward) to a hidden-state edit
delta, trained with MSE against the feedback-derived target delta. The
judge, feedback queue, and per-instance lifecycle are plain Python in
``feedback.py``/``consumer.py``; execution (lanes, functional optimizer
transitions, CUDA graph capture) is derived by ``tllm.program.compile_program``.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from tllm.consumers._common.alignment import LowRankAdapter
from tllm.ports.base import TimingDeadline
from tllm.workgraph.frontend import extern, flow, local, provided, result


class FTTTProgram(torch.nn.Module):
    """Predict a hidden-state edit delta from an edit input row."""

    def __init__(
        self,
        *,
        hidden_size: int,
        feedback_feature_size: int,
        rank: int,
        lr: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        edit_input_size = int(hidden_size) + max(1, int(feedback_feature_size)) + 1
        self.adapter = LowRankAdapter(
            in_dim=edit_input_size, out_dim=int(hidden_size), rank=rank, dtype=dtype, device=device
        )
        self.optimizer = torch.optim.Adam(
            self.adapter.parameters(), lr=lr, foreach=False
        )

    @flow(
        inputs={"edit_input": provided(), "target": provided()},
        outputs={"loss": local()},
        state="edit_adapter",
    )
    def forward(self, edit_input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.adapter(edit_input), target)

    @flow(
        inputs={"edit_input": provided()},
        outputs={"delta": local()},
        state="edit_adapter",
    )
    def predict(self, edit_input: torch.Tensor) -> torch.Tensor:
        # Read-only invocation (ProgramConsumer.run_flow); the PG300
        # "unconsumed local" diagnostic is expected.
        return self.adapter(edit_input)

    @extern(
        inputs={"loss": result("forward", "loss")},
        after=("forward",),
        state="edit_adapter",
        deadline=TimingDeadline.NEXT_PREPARE,
    )
    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()


__all__ = ["FTTTProgram"]

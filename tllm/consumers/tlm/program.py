#!/usr/bin/env python3
"""TLM expressed as a tLLM Program.

The whole algorithm: a low-rank adapter mapping hidden rows to vocab deltas,
trained with the upstream-TTL loss — per-row sentence CE against the base
logits selects hard rows (CE > threshold) and weights them by
``lamb * exp(ce - threshold)`` on the adapted logits' one-hot CE. Row
selection is expressed as a fixed-shape masked mean so the step stays one
static DAG (CUDA-graph capturable); a step with no selected rows contributes
zero loss and zero gradient.
"""

from __future__ import annotations

import torch

from tllm.consumers._common.alignment import LowRankAdapter, sentence_cross_entropy
from tllm.ports.base import TimingDeadline
from tllm.workgraph.frontend import extern, flow, local, provided, result


class TLMProgram(torch.nn.Module):
    """Adapter(hidden) -> vocab delta, trained on hard rows only."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        lr: float,
        threshold: float,
        lamb: float,
        ignore_index: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.adapter = LowRankAdapter(
            in_dim=hidden_size, out_dim=vocab_size, rank=rank, dtype=dtype, device=device
        )
        self.optimizer = torch.optim.Adam(
            self.adapter.parameters(), lr=lr, foreach=False
        )
        self.threshold = float(threshold)
        self.lamb = float(lamb)
        self.ignore_index = int(ignore_index)

    @flow(
        inputs={
            "hidden": provided(),
            "logits": provided(),
            "labels": provided(),
            "base_logits": provided(),
        },
        outputs={"loss": local()},
        state="adapters",
    )
    def forward(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        sentence_ce = sentence_cross_entropy(
            base_logits, labels, ignore_index=self.ignore_index
        )
        adapted = logits + self.adapter(hidden)
        ce = sentence_cross_entropy(adapted, labels, ignore_index=self.ignore_index)
        selected = (sentence_ce > self.threshold).to(ce.dtype)
        weights = (
            self.lamb * torch.exp((sentence_ce - self.threshold).clamp_min(0.0)) * selected
        )
        return (ce * weights).sum() / selected.sum().clamp_min(1.0)

    @flow(
        inputs={"hidden": provided(), "logits": provided()},
        outputs={"adapted_logits": local()},
        state="adapters",
    )
    def predict(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        # Read-only invocation (ProgramConsumer.run_flow) against the current
        # bank state; the PG300 "unconsumed local" diagnostic is expected.
        return logits + self.adapter(hidden)

    @extern(
        inputs={"loss": result("forward", "loss")},
        after=("forward",),
        state="adapters",
        deadline=TimingDeadline.NEXT_PREPARE,
    )
    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()


__all__ = ["TLMProgram"]

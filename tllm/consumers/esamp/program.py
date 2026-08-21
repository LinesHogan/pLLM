#!/usr/bin/env python3
"""ESamp expressed as a tLLM Program.

This is the entire algorithm: a low-rank gated residual model, one flow
(predict + loss on the residual pair), one extern (optimizer step). All
scheduling, state banking, and sampler wiring are derived from the
declarations by ``tllm.program.compile_program`` — there is no metadata,
stream, buffer, graph, or lease handling here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tllm.consumers.esamp.model import LowRankGatedResidualModel
from tllm.ports.base import TimingDeadline
from tllm.workgraph.frontend import extern, flow, hidden, local, result, sink


class ESampProgram(torch.nn.Module):
    """Predict the final-layer hidden state from the first-layer one."""

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
        self.model = LowRankGatedResidualModel(
            hidden_size=hidden_size,
            rank=rank,
            use_output_layernorm=True,
            device=device,
            dtype=dtype,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, foreach=False
        )

    @flow(
        inputs={
            "source": hidden(layer=0, site="block_output", phase="decode"),
            "target": hidden(layer=-1, site="block_output", phase="decode"),
        },
        outputs={"prediction": sink("guidance"), "loss": local()},
        state="distiller",
    )
    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self.model(source)
        return prediction, F.mse_loss(prediction.float(), target.float())

    @flow(
        inputs={
            "source": hidden(layer=0, site="block_output", phase="decode"),
        },
        outputs={"prediction": local()},
        state="distiller",
    )
    def predict(self, source: torch.Tensor) -> torch.Tensor:
        # Invoked on demand by the sampler provider (ProgramConsumer.run_flow)
        # against the current bank state, so the PG300 "unconsumed local"
        # diagnostic on this output is expected.
        return self.model(source)

    @extern(
        inputs={"loss": result("forward", "loss")},
        after=("forward",),
        state="distiller",
        deadline=TimingDeadline.NEXT_PREPARE,
    )
    def backward(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()


__all__ = ["ESampProgram"]
